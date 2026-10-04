"""Ask Cosmos-Reason2 on the probe endpoint to describe the driveway photo.

Uses the OpenAI SDK against {COSMOS_URL}/v1 with COSMOS_TOKEN, both read from
.env where scripts/cosmos_up.py saved them. Each image is sent as a base64
data URL, after applying its EXIF orientation so the picture is upright and
resizing it to 1280 px on the long side.

  Test A (car present): the full photo.
          PASS if the JSON reply has vehicle_present true.
  Test B (no car): the bottom 40% of the same upright photo, which shows only
          empty asphalt. PASS if vehicle_present is false.
  Test C (free text): the full photo. PASS if a description comes back.

For each test prints the raw reply, any reasoning text, latency, token usage
and a PASS/FAIL line. Exits non-zero if a test fails.

    .venv/bin/python scripts/cosmos_test.py [--model nvidia/Cosmos-Reason2-2B]

The photo shows a real licence plate: samples/ is gitignored, and the replies
are only printed, never written to a file.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
from dotenv import dotenv_values
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
PHOTO = REPO_ROOT / "samples" / "IMG_2457.jpg"
DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"
LONG_SIDE = 1280
MAX_TOKENS = 2048

JSON_QUESTION = (
    "This is a fixed camera view of a residential driveway. Answer in JSON with "
    "keys vehicle_present (true/false), vehicle_description, position "
    "(near/middle/far and left/centre/right), blocking_anything (true/false, and "
    "what), confidence (0-1)."
)
FREE_TEXT_QUESTION = (
    "Describe what a parking attendant would need to know about this scene in "
    "two sentences."
)


def to_data_url(image) -> tuple[str, str]:
    """Resize to 1280 px on the long side and encode as a base64 JPEG data URL.

    Returns the data URL and the size that was sent, as "WIDTHxHEIGHT".
    """
    height, width = image.shape[:2]
    scale = LONG_SIDE / max(height, width)
    if scale < 1:
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode the image as JPEG")
    encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", f"{image.shape[1]}x{image.shape[0]}"


def vehicle_present(reply: str) -> bool | None:
    """vehicle_present from the JSON object in the reply; None if it has none.

    Tolerates text or a code fence around the object, and "true"/"false" strings.
    """
    start, end = reply.find("{"), reply.rfind("}")
    try:
        value = json.loads(reply[start : end + 1]).get("vehicle_present")
    except (ValueError, AttributeError):
        return None
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value if isinstance(value, bool) else None


def run_test(client: OpenAI, model: str, name: str, image, question: str, expected) -> bool:
    """Ask one question about one image, print everything, and return whether it passed.

    expected is the vehicle_present value required to pass, or None for free text.
    """
    data_url, size = to_data_url(image)
    print(f"\n--- {name} | image sent: {size} ---")
    print(f"question: {question}")
    try:
        start = time.perf_counter()
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        latency = time.perf_counter() - start
        body = json.loads(raw.text)
        choice = body["choices"][0]
        message = choice["message"]
        reply = message.get("content") or ""
        # vLLM's reasoning parser puts any <think> text in a separate field.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        print(f"raw reply:\n{reply}")
        print(f"reasoning:\n{reasoning or '(none)'}")
        print(f"finish_reason: {choice.get('finish_reason')}")
        print(f"latency: {latency:.2f}s")
        print(f"usage:   {json.dumps(body.get('usage'))}")
        if expected is None:
            passed, reason = bool(reply.strip()), "empty reply"
        else:
            found = vehicle_present(reply)
            passed = found is expected
            reason = (
                "no JSON object with a true/false vehicle_present in the reply"
                if found is None
                else f"vehicle_present is {str(found).lower()}"
            )
    except Exception as err:  # API errors, timeouts, malformed responses
        passed = False
        reason = f"{type(err).__name__}: {err}".replace(str(client.api_key), "[REDACTED]")
    print(f"{'PASS' if passed else 'FAIL'} | {name}" + ("" if passed else f" | {reason}"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask Cosmos-Reason2 on the probe endpoint about the driveway photo."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model the endpoint serves (default {DEFAULT_MODEL})",
    )
    model = parser.parse_args().model

    env = dotenv_values(ENV_PATH)
    url = (env.get("COSMOS_URL") or "").strip().rstrip("/")
    token = (env.get("COSMOS_TOKEN") or "").strip()
    if not url or not token:
        print(
            "COSMOS_URL or COSMOS_TOKEN is missing from .env: run scripts/cosmos_up.py first",
            file=sys.stderr,
        )
        return 2

    # IMREAD_COLOR applies the EXIF orientation, so the frame comes back upright.
    upright = cv2.imread(str(PHOTO), cv2.IMREAD_COLOR)
    if upright is None:
        print(f"could not read {PHOTO}", file=sys.stderr)
        return 2
    height, width = upright.shape[:2]
    bottom = upright[round(height * 0.6) :]
    print(f"endpoint: {url}/v1 | model: {model}")
    print(f"photo: samples/{PHOTO.name}, upright {width}x{height}")

    client = OpenAI(api_key=token, base_url=f"{url}/v1", timeout=300, max_retries=0)
    results = [
        run_test(client, model, "Test A (car present)", upright, JSON_QUESTION, True),
        run_test(client, model, "Test B (no car, bottom 40%)", bottom, JSON_QUESTION, False),
        run_test(client, model, "Test C (free text)", upright, FREE_TEXT_QUESTION, None),
    ]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
