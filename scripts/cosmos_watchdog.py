"""Delete the spotter-cosmos endpoint after a time limit, no matter what else happens.

    .venv/bin/python scripts/cosmos_watchdog.py --hours 3

Runs in the background from the moment the endpoint is started. When the
limit is reached it runs scripts/cosmos_down.py, which deletes every endpoint
named spotter-cosmos and proves none remains. If the endpoint is already gone
when the limit is reached, cosmos_down.py simply reports that. The Nebius
endpoint API has no time limit or automatic stop of its own (checked in
`nebius ai endpoint create --help`), so this is the automatic stop.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete the Cosmos endpoint after a time limit.")
    parser.add_argument("--hours", type=float, default=3.0)
    args = parser.parse_args()
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    print(f"watchdog: will delete the endpoint at {deadline.isoformat(timespec='seconds')} ({args.hours:g} h from now)", flush=True)
    while datetime.now(timezone.utc) < deadline:
        time.sleep(30)
    print("watchdog: time limit reached, running scripts/cosmos_down.py", flush=True)
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "cosmos_down.py")], cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
