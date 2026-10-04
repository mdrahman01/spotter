"""Start the Cosmos-Reason2 probe endpoint on Nebius and wait for the model to load.

Creates a Serverless AI endpoint named spotter-cosmos: vLLM serving
nvidia/Cosmos-Reason2-8B on one L40S GPU, behind a random access token. Then
waits in two stages, because running is not the same as loaded:

  1. until `nebius ai endpoint get` reports RUNNING with an https URL;
  2. until GET {url}/v1/models returns 200 (the model download and load can
     take 10 to 20 minutes).

    .venv/bin/python scripts/cosmos_up.py [--model nvidia/Cosmos-Reason2-2B]

Reads HF_TOKEN from .env and saves COSMOS_TOKEN and COSMOS_URL there. Secrets
are never printed. Before creating anything it checks that HF_TOKEN can read
the gated model (HTTP 200 on its config.json) and refuses to start otherwise.
After 30 minutes it gives up, prints the last 100 lines of the endpoint's logs
and exits non-zero.

The endpoint is billed for as long as it exists, loaded or not: always run
scripts/cosmos_down.py afterwards.
"""

import argparse
import json
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values, set_key

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
# The installer puts the CLI in ~/.nebius/bin, which is on PATH only in
# interactive shells.
NEBIUS = shutil.which("nebius") or str(Path.home() / ".nebius" / "bin" / "nebius")

ENDPOINT_NAME = "spotter-cosmos"
IMAGE = "vllm/vllm-openai:v0.18.0-cu130"
DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"
PLATFORM = "gpu-l40s-a"
PRESET = "1gpu-8vcpu-32gb"
PORT = 8000

GIVE_UP_SECONDS = 30 * 60
POLL_SECONDS = 10
HEARTBEAT_SECONDS = 180

EXIT_FAILED = 1
EXIT_PREREQUISITE = 2

START = time.monotonic()
SECRETS: list[str] = []  # every value that must never reach the output


def redact(text: str) -> str:
    for secret in SECRETS:
        text = text.replace(secret, "[REDACTED]")
    return text


def log(message: str) -> None:
    """Print one timestamped line: UTC time and seconds since the script started."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp} +{time.monotonic() - START:.0f}s] {redact(message)}", flush=True)


def nebius(*args: str) -> subprocess.CompletedProcess:
    """Run the Nebius CLI and capture its output. The command line is never printed."""
    return subprocess.run(
        [NEBIUS, *args, "--no-progress"], capture_output=True, text=True, timeout=120
    )


def cli_error(result: subprocess.CompletedProcess) -> str:
    """The last lines of a failed CLI call, on one line."""
    lines = (result.stderr or result.stdout).strip().splitlines()
    return " | ".join(lines[-3:]) or f"exit code {result.returncode}"


def existing_endpoints() -> list[dict]:
    """Endpoints named spotter-cosmos in the CLI profile's project."""
    result = nebius("ai", "endpoint", "list", "--format", "json")
    if result.returncode != 0:
        raise RuntimeError(f"`nebius ai endpoint list` failed: {cli_error(result)}")
    items = (json.loads(result.stdout or "{}") or {}).get("items") or []
    return [e for e in items if e.get("metadata", {}).get("name") == ENDPOINT_NAME]


def created_id(stdout: str) -> str | None:
    """The new endpoint's ID: resource_id of the create operation, else looked up by name."""
    try:
        return json.loads(stdout)["resource_id"]
    except (ValueError, KeyError, TypeError):
        pass
    try:
        found = existing_endpoints()
    except (RuntimeError, subprocess.TimeoutExpired, ValueError):
        return None
    return found[0]["metadata"]["id"] if found else None


def endpoint_status(endpoint_id: str) -> dict:
    """The endpoint's status block; empty if the CLI call failed, which is treated as transient."""
    try:
        result = nebius("ai", "endpoint", "get", "--id", endpoint_id, "--format", "json")
        if result.returncode != 0:
            log(f"`nebius ai endpoint get` failed, will retry: {cli_error(result)}")
            return {}
        return json.loads(result.stdout).get("status") or {}
    except (subprocess.TimeoutExpired, ValueError) as err:
        log(f"`nebius ai endpoint get` failed, will retry: {type(err).__name__}")
        return {}


def models_answer(url: str, token: str) -> int | str:
    """HTTP status of GET {url}/v1/models with the bearer token, or a short error."""
    request = urllib.request.Request(
        f"{url}/v1/models", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as err:
        return err.code
    except OSError as err:  # URLError, timeouts, connection resets
        return f"{type(err).__name__}: {getattr(err, 'reason', err)}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect, so HF_TOKEN is only ever sent to huggingface.co."""

    def redirect_request(self, *args, **kwargs):
        return None


def model_access(model: str, hf_token: str) -> int | str:
    """HTTP status HF_TOKEN gets for the gated model's config.json, or a short error.

    200 means the token can download the model. 403 means its account has not
    been granted access to the gated repo, and vLLM would fail at start-up on
    an endpoint that is already being billed.
    """
    request = urllib.request.Request(
        f"https://huggingface.co/{model}/resolve/main/config.json",
        headers={"Authorization": f"Bearer {hf_token}"},
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as err:
        return err.code
    except OSError as err:  # URLError, timeouts, connection resets
        return f"{type(err).__name__}: {getattr(err, 'reason', err)}"


def give_up(endpoint_id: str, reason: str) -> int:
    """Report a failed start with the last 100 log lines. The endpoint still exists."""
    log(f"giving up: {reason}")
    log(f"last 100 lines of `nebius ai endpoint logs {endpoint_id}`:")
    try:
        result = nebius("ai", "endpoint", "logs", endpoint_id, "--tail", "100")
        output = (result.stdout + result.stderr).strip() or "(no log output)"
    except subprocess.TimeoutExpired:
        output = "(the logs command timed out)"
    print(redact(output), flush=True)
    log("the endpoint still exists and is billed: run scripts/cosmos_down.py")
    return EXIT_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the spotter-cosmos endpoint and wait for the model to load."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model for vLLM to serve (default {DEFAULT_MODEL})",
    )
    model = parser.parse_args().model

    hf_token = (dotenv_values(ENV_PATH).get("HF_TOKEN") or "").strip()
    if not hf_token:
        log("HF_TOKEN is missing from .env")
        return EXIT_PREREQUISITE
    SECRETS.append(hf_token)
    access = model_access(model, hf_token)
    if access != 200:
        log(
            f"HF_TOKEN cannot read {model} (config.json -> HTTP {access}): no endpoint "
            f"created. Request access at https://huggingface.co/{model}"
        )
        return EXIT_PREREQUISITE
    log(f"HF_TOKEN can read {model} (config.json -> HTTP 200)")
    try:
        already_there = existing_endpoints()
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as err:
        log(f"could not list endpoints: {err}")
        return EXIT_PREREQUISITE
    if already_there:
        log(f"an endpoint named {ENDPOINT_NAME} already exists: run scripts/cosmos_down.py first")
        return EXIT_PREREQUISITE

    token = secrets.token_hex(32)
    SECRETS.append(token)
    set_key(ENV_PATH, "COSMOS_TOKEN", token, quote_mode="never")
    log("saved a new random 64-hex COSMOS_TOKEN to .env")

    serve = (
        f"serve {model} --host 0.0.0.0 --port {PORT} "
        "--max-model-len 16384 --reasoning-parser qwen3"
    )
    log(
        f"creating endpoint {ENDPOINT_NAME}: image {IMAGE}, command `vllm {serve}`, "
        f"{PLATFORM} {PRESET}, on-demand, no public IP, token auth"
    )
    # The CLI takes the token and HF_TOKEN only as arguments (or as MysteryBox
    # secrets), so they are passed here and nowhere printed.
    try:
        result = nebius(
            "ai", "endpoint", "create",
            "--name", ENDPOINT_NAME,
            "--image", IMAGE,
            "--container-command", "vllm",
            "--args", serve,
            "--platform", PLATFORM,
            "--preset", PRESET,
            "--container-port", str(PORT),
            "--auth", "token",
            "--token", token,
            "--env", f"HF_TOKEN={hf_token}",
            "--shm-size", "16Gi",
            "--on-demand",  # a regular VM, not a preemptible one
            # No --public: the https URL is managed by Nebius and needs no public IP.
            "--async",
            "--format", "json",
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        log("create did not return; an endpoint may exist: run scripts/cosmos_down.py")
        return EXIT_FAILED
    if result.returncode != 0:
        log(f"create failed: {cli_error(result)}")
        return EXIT_FAILED
    endpoint_id = created_id(result.stdout)
    if not endpoint_id:
        log("create succeeded but the new endpoint could not be found")
        return EXIT_FAILED
    log(f"create accepted: endpoint {endpoint_id}")

    # Stage 1: wait for RUNNING with an https URL.
    stages = [("create call", START)]  # (stage name, time it began)
    state = url = None
    while True:
        status = endpoint_status(endpoint_id)
        if status.get("state") and status["state"] != state:
            state = status["state"]
            stages.append((state, time.monotonic()))
            log(f"endpoint state: {state}")
        url = next(
            (u for u in status.get("public_endpoints") or [] if u.startswith("https://")),
            None,
        )
        if state == "RUNNING" and url:
            break
        if state in ("ERROR", "STOPPED"):
            details = json.dumps(status.get("state_details"))
            return give_up(endpoint_id, f"endpoint state is {state}: {details}")
        if time.monotonic() - START > GIVE_UP_SECONDS:
            return give_up(endpoint_id, f"not running after 30 minutes (state {state})")
        time.sleep(POLL_SECONDS)

    url = url.rstrip("/")
    set_key(ENV_PATH, "COSMOS_URL", url, quote_mode="never")
    log(f"endpoint is RUNNING at {url}; saved to .env as COSMOS_URL")

    # Stage 2: wait for vLLM to download and load the model.
    log(f"waiting for the model to load: polling GET {url}/v1/models")
    answer = last_logged = None
    last_logged_at = 0.0
    while True:
        answer = models_answer(url, token)
        if answer == 200:
            break
        if answer != last_logged or time.monotonic() - last_logged_at > HEARTBEAT_SECONDS:
            log(f"/v1/models -> {answer}")
            last_logged, last_logged_at = answer, time.monotonic()
        new_state = endpoint_status(endpoint_id).get("state")
        if new_state and new_state != state:
            state = new_state
            log(f"endpoint state: {state}")
        if state in ("ERROR", "STOPPED"):
            return give_up(endpoint_id, f"endpoint state became {state} while loading")
        if time.monotonic() - START > GIVE_UP_SECONDS:
            return give_up(
                endpoint_id, f"model not loaded after 30 minutes (/v1/models -> {answer})"
            )
        time.sleep(POLL_SECONDS)
    log(f"/v1/models -> 200: {model} is loaded")

    stages.append(("loaded", time.monotonic()))
    spans = [
        f"{'RUNNING until model loaded' if name == 'RUNNING' else name} {end - begin:.0f}s"
        for (name, begin), (_, end) in zip(stages, stages[1:])
    ]
    log(f"stage times: {' | '.join(spans)} | total {stages[-1][1] - START:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
