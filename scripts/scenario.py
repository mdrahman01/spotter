"""Set up and run one named scenario end to end, then check it.

    .venv/bin/python scripts/scenario.py overstay --expect PLATE
    SPOTTER_ALERTS=telegram .venv/bin/python scripts/scenario.py overstay --expect PLATE
    .venv/bin/python scripts/scenario.py booked --expect PLATE
    .venv/bin/python scripts/scenario.py unknown --expect PLATE

Each scenario empties the database, seeds it, replays samples/IMG_2472.mov
through spotter.run, and ends with scripts/check_attendant.py, whose PASS or
FAIL line is the last thing printed. The plate is passed on to the commands
and never printed. SPOTTER_* settings in the environment reach the run.

  overstay  booking from 60 s before the replay base time to 12 s after it,
            ending_soon_s 4, overstay_grace_s 3: ARRIVED at about 7 s, then
            ENDING_SOON at about 8 s, OVERSTAY at about 15 s, LEFT at about 26 s.
  booked    a 12-hour booking that starts now.
  unknown   no booking.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
VIDEO = "samples/IMG_2472.mov"
DRIVER = "Test Driver"
RATE = "5.00"
SCENARIOS = ("overstay", "booked", "unknown")


def run(step: str, command: list[str], plate: str) -> int:
    shown = " ".join("<plate>" if part == plate else part for part in command)
    print(f"\n$ {shown}  [{step}]", flush=True)
    code = subprocess.run(command, cwd=REPO_ROOT).returncode
    print(f"[{step} exit: {code}]", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one attendant scenario end to end and check it.")
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--expect", required=True, metavar="PLATE", help="your plate text")
    args = parser.parse_args()
    plate = args.expect
    seed = [PYTHON, "-m", "spotter.seed", "--reset"]
    watch = [PYTHON, "-m", "spotter.run", "--source", VIDEO, "--expect", plate]

    if args.scenario == "overstay":
        base = datetime.now(timezone.utc).replace(microsecond=0)
        seed += [
            "--plate", plate, "--driver", DRIVER, "--rate", RATE,
            "--starts-at", (base - timedelta(seconds=60)).isoformat(),
            "--ends-at", (base + timedelta(seconds=12)).isoformat(),
        ]
        watch += ["--replay-base-time", base.isoformat(), "--ending-soon-s", "4", "--overstay-grace-s", "3"]
    elif args.scenario == "booked":
        seed += ["--plate", plate, "--driver", DRIVER, "--hours", "12", "--rate", RATE]

    print(f"######## SCENARIO {args.scenario} ########")
    (REPO_ROOT / "data" / "events.jsonl").unlink(missing_ok=True)
    if run("seed", seed, plate) != 0:
        return 1
    run("run", watch, plate)
    return run("check", [PYTHON, "scripts/check_attendant.py", args.scenario], plate)


if __name__ == "__main__":
    sys.exit(main())
