"""Hello-world check that NVIDIA Nemotron on Nebius Token Factory works for us.

For each model in MODELS:
  a) plain test: ask for an exact reply; print the reply, latency and token usage.
  b) tool test:  offer one tool, lookup_booking(plate), and expect the model to
     call it with plate "ABC1234"; then send back a fake tool result and print
     the model's final answer.

Prints one PASS/FAIL line per model per test.

Run from anywhere, with NEBIUS_API_KEY set in the repo's .env:

    .venv/bin/python scripts/hello_nemotron.py

Exit codes: 0 = all tests passed, 1 = a test failed, 2 = NEBIUS_API_KEY missing,
3 = a model ID was rejected (the NVIDIA models from client.models.list() are
printed before exiting).
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODELS = [
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]

PLAIN_PROMPT = "Reply with exactly: Spotter online."
PLAIN_EXPECTED = "Spotter online."

TOOL_PROMPT = (
    "A car with plate ABC1234 just pulled into the driveway. "
    "Check whether it has a booking."
)
TOOL_NAME = "lookup_booking"
EXPECTED_PLATE = "ABC1234"
FAKE_TOOL_RESULT = {"booked": True, "driver": "Test Driver", "until": "07:00"}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Look up whether a vehicle has a booking, by its licence plate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plate": {
                        "type": "string",
                        "description": "Licence plate of the vehicle.",
                    }
                },
                "required": ["plate"],
            },
        },
    }
]

EXIT_TEST_FAILED = 1
EXIT_NO_API_KEY = 2
EXIT_MODEL_REJECTED = 3


def safe_error(client: OpenAI, err: Exception) -> str:
    """Describe an error for printing, with the API key scrubbed out."""
    return f"{type(err).__name__}: {err}".replace(str(client.api_key), "[REDACTED]")


def chat(client: OpenAI, **request):
    """Run one chat completion.

    Returns (completion, body, latency): the parsed completion, the response
    body exactly as received, and the wall-clock seconds the request took.
    """
    start = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(**request)
    latency = time.perf_counter() - start
    completion = raw.parse()
    if not completion.choices:
        raise RuntimeError("response contained no choices")
    return completion, json.loads(raw.text), latency


def is_booking_call(call) -> bool:
    """True for a lookup_booking call whose arguments include plate "ABC1234"."""
    if call.function.name != TOOL_NAME:
        return False
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, ValueError):
        return False
    return isinstance(arguments, dict) and arguments.get("plate") == EXPECTED_PLATE


def plain_test(client: OpenAI, model: str) -> tuple[bool, str]:
    """PASS if the reply, ignoring surrounding whitespace, is exactly PLAIN_EXPECTED."""
    print(f"prompt:  {PLAIN_PROMPT}")
    completion, body, latency = chat(
        client,
        model=model,
        messages=[{"role": "user", "content": PLAIN_PROMPT}],
        temperature=0,
    )
    reply = completion.choices[0].message.content or ""
    print(f"reply:   {reply!r}")
    print(f"latency: {latency:.2f}s")
    print(f"usage:   {json.dumps(body.get('usage'))}")
    if reply.strip() != PLAIN_EXPECTED:
        return False, f"reply is not exactly {PLAIN_EXPECTED!r}"
    return True, ""


def tool_test(client: OpenAI, model: str) -> tuple[bool, str]:
    """PASS if the model calls lookup_booking with plate "ABC1234" and then,
    given the fake tool result, produces a final answer."""
    print(f"prompt:  {TOOL_PROMPT}")
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    completion, body, latency = chat(
        client, model=model, messages=messages, tools=TOOLS, temperature=0
    )
    message = completion.choices[0].message
    tool_calls = message.tool_calls or []
    print(f"latency: {latency:.2f}s (tool-call turn)")
    print("raw tool_calls JSON:")
    print(json.dumps(body["choices"][0]["message"].get("tool_calls"), indent=2))

    if not tool_calls:
        print(f"reply instead of a tool call: {message.content!r}")
        return False, "response contains no tool call"
    if not any(is_booking_call(call) for call in tool_calls):
        return False, f'no {TOOL_NAME} call with plate "{EXPECTED_PLATE}"'

    # Answer every tool call, so the follow-up request is valid even if the
    # model made more than one.
    messages.append(
        {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }
    )
    tool_result = json.dumps(FAKE_TOOL_RESULT)
    for call in tool_calls:
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": tool_result}
        )
    print(f"tool result sent back: {tool_result}")
    final, _, latency = chat(
        client, model=model, messages=messages, tools=TOOLS, temperature=0
    )
    answer = (final.choices[0].message.content or "").strip()
    print(f"latency: {latency:.2f}s (final-answer turn)")
    print(f"final answer: {answer}")
    if not answer:
        return False, "no final answer after the tool result"
    return True, ""


def stop_if_model_rejected(client: OpenAI, model: str, err: APIStatusError) -> None:
    """If the API refused the model ID itself, show the NVIDIA models and stop.

    A 404 always counts. Any other 4xx counts only when client.models.list()
    does not know the ID; otherwise the request, not the model ID, was the
    problem and the caller records an ordinary test failure.
    """
    if err.status_code not in (400, 403, 404, 422):
        return
    try:
        model_ids = sorted(m.id for m in client.models.list())
    except APIError as list_err:
        if err.status_code != 404:
            return
        print(f"MODEL ID REJECTED | {model} | {safe_error(client, err)}")
        print(f"client.models.list() also failed | {safe_error(client, list_err)}")
        sys.exit(EXIT_MODEL_REJECTED)
    if err.status_code != 404 and model in model_ids:
        return
    print(f"MODEL ID REJECTED | {model} | {safe_error(client, err)}")
    print("NVIDIA models returned by client.models.list():")
    nvidia = [m for m in model_ids if "nvidia" in m.lower() or "nemotron" in m.lower()]
    for model_id in nvidia:
        print(f"  {model_id}")
    if not nvidia:
        print(f"  (none among the {len(model_ids)} models listed)")
    sys.exit(EXIT_MODEL_REJECTED)


def run_test(client: OpenAI, model: str, name: str, test) -> bool:
    """Run one test, print its PASS/FAIL line, and return whether it passed."""
    print(f"\n--- {model} | {name} test ---")
    try:
        passed, reason = test(client, model)
    except APIStatusError as err:
        stop_if_model_rejected(client, model, err)
        passed, reason = False, safe_error(client, err)
    except Exception as err:  # timeouts, connection errors, malformed responses
        passed, reason = False, safe_error(client, err)
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict} | {model} | {name} test" + (f" | {reason}" if reason else ""))
    return passed


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        print(
            "NEBIUS_API_KEY is missing. Add it to .env (see .env.example).",
            file=sys.stderr,
        )
        return EXIT_NO_API_KEY

    # No SDK retries: latency numbers stay honest and every error is visible.
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120, max_retries=0)
    print(f"base_url: {BASE_URL}")

    results = [
        run_test(client, model, name, test)
        for model in MODELS
        for name, test in (("plain", plain_test), ("tool", tool_test))
    ]
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} tests passed")
    return EXIT_TEST_FAILED if failed else 0


if __name__ == "__main__":
    sys.exit(main())
