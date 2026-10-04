"""Delete the Cosmos-Reason2 probe endpoint and prove that none remains.

Deletes every endpoint named spotter-cosmos in the CLI profile's project,
waits until it is gone, then lists the endpoints that are left. Safe to run
when nothing exists.

    .venv/bin/python scripts/cosmos_down.py

Also prints how long each deleted endpoint existed and the estimated cost at
the L40S on-demand price. Exits non-zero if spotter-cosmos still exists at the
end.
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
# The installer puts the CLI in ~/.nebius/bin, which is on PATH only in
# interactive shells.
NEBIUS = shutil.which("nebius") or str(Path.home() / ".nebius" / "bin" / "nebius")

ENDPOINT_NAME = "spotter-cosmos"
PRICE_PER_HOUR = 1.59  # USD, gpu-l40s-a 1gpu-8vcpu-32gb on demand
DELETE_TIMEOUT_SECONDS = 10 * 60
GONE_WAIT_SECONDS = 15 * 60
POLL_SECONDS = 10

# Every .env value except the URL is treated as a secret and never printed.
SECRETS = [v for k, v in dotenv_values(ENV_PATH).items() if v and k != "COSMOS_URL"]


def log(message: str) -> None:
    """Print one timestamped line, with any secret scrubbed out."""
    for secret in SECRETS:
        message = message.replace(secret, "[REDACTED]")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def nebius(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run the Nebius CLI and capture its output."""
    return subprocess.run(
        [NEBIUS, *args, "--no-progress"], capture_output=True, text=True, timeout=timeout
    )


def cli_error(result: subprocess.CompletedProcess) -> str:
    """The last lines of a failed CLI call, on one line."""
    lines = (result.stderr or result.stdout).strip().splitlines()
    return " | ".join(lines[-3:]) or f"exit code {result.returncode}"


def list_endpoints() -> list[dict]:
    """Every endpoint in the CLI profile's project."""
    result = nebius("ai", "endpoint", "list", "--format", "json")
    if result.returncode != 0:
        raise RuntimeError(f"`nebius ai endpoint list` failed: {cli_error(result)}")
    return (json.loads(result.stdout or "{}") or {}).get("items") or []


def describe(endpoint: dict) -> str:
    """Name, ID and state only: the spec can hold secrets and is never printed."""
    metadata = endpoint.get("metadata", {})
    state = endpoint.get("status", {}).get("state")
    return f"{metadata.get('name')} | {metadata.get('id')} | {state}"


def is_ours(endpoint: dict) -> bool:
    return endpoint.get("metadata", {}).get("name") == ENDPOINT_NAME


def minutes_existed(endpoint: dict, until: datetime) -> float | None:
    """Minutes from the endpoint's created_at to `until`; None if created_at is unusable."""
    try:
        created = datetime.fromisoformat(endpoint["metadata"]["created_at"])
        return (until - created).total_seconds() / 60
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    try:
        targets = [e for e in list_endpoints() if is_ours(e)]
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as err:
        log(f"could not list endpoints, so nothing is proven: {err}")
        return 1

    if not targets:
        log(f"no endpoint named {ENDPOINT_NAME} exists: nothing to delete")
    for endpoint in targets:
        endpoint_id = endpoint["metadata"]["id"]
        log(f"deleting {describe(endpoint)}")
        try:
            result = nebius(
                "ai", "endpoint", "delete", "--id", endpoint_id, timeout=DELETE_TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                log(f"delete of {endpoint_id} finished")
            else:
                log(f"delete of {endpoint_id} failed: {cli_error(result)}")
        except subprocess.TimeoutExpired:
            log(f"delete of {endpoint_id} did not return within {DELETE_TIMEOUT_SECONDS}s")

    # Wait until the project no longer lists the endpoint, whatever delete said.
    deadline = time.monotonic() + GONE_WAIT_SECONDS
    endpoints, remaining, last_seen = None, targets, None
    while True:
        try:
            endpoints = list_endpoints()
            remaining = [e for e in endpoints if is_ours(e)]
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as err:
            log(f"could not list endpoints, will retry: {err}")
        else:
            if not remaining:
                break
            seen = "; ".join(describe(e) for e in remaining)
            if seen != last_seen:
                log(f"still listed: {seen}")
                last_seen = seen
        if time.monotonic() > deadline:
            break
        time.sleep(POLL_SECONDS)

    gone_at = datetime.now(timezone.utc)
    if not remaining:
        for endpoint in targets:
            minutes = minutes_existed(endpoint, gone_at)
            if minutes is None:
                log(f"{endpoint['metadata']['id']} is gone (no created_at to time it by)")
                continue
            log(
                f"{endpoint['metadata']['id']} existed for {minutes:.1f} minutes: "
                f"about ${minutes / 60 * PRICE_PER_HOUR:.2f} at ${PRICE_PER_HOUR}/hour"
            )

    if endpoints is None:
        log("could not list endpoints, so nothing is proven")
        return 1
    log(f"endpoints left in the project: {len(endpoints)}")
    for endpoint in endpoints:
        log(f"  {describe(endpoint)}")
    if remaining:
        log(f"{ENDPOINT_NAME} STILL EXISTS and is billed: delete it by hand")
        return 1
    log(f"no endpoint named {ENDPOINT_NAME} remains")
    return 0


if __name__ == "__main__":
    sys.exit(main())
