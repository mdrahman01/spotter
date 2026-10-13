"""Replay the sample clip with scripted scene reports and check what the scene changed.

    .venv/bin/python scripts/scene_replays.py --expect PLATE [early|held|unreadable ...]

No endpoint is needed: SPOTTER_SCENE=script:<file> plays canned reports. The
clip shows the car from about 5.8 s to 22.2 s of 30.4 s. Three replays:

  early       the scene sees an empty spot once the plate is unread: LEFT is
              confirmed by the scene before leave_after_s (6 s here) runs out.
  held        plate reads are dropped from 10 s to 20 s (SPOTTER_TEST_DROP_READS)
              while the scene still sees the car: no LEFT at 14 s, the stay is
              held, the plate is read again at 20 s, and the LEFT comes after
              the real departure at 22.2 s, confirmed by the scene.
  unreadable  every plate read is dropped and the scene sees a vehicle from
              6 s to 22 s: UNREADABLE_VEHICLE, then SPOT_CLEAR, and no stay.

Each replay runs spotter.run with console alerts and the console light, then
reads data/events.jsonl and, for the unreadable case, runs the attendant check.
Prints PASS or FAIL per replay. Prints no plate text.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
VIDEO = "samples/IMG_2472.mov"
CLIP_ZONE = "0.35,0.27,0.69,0.69"
EMPTY = {"vehicle_in_driveway": False, "plate_visible": False, "what": "empty asphalt"}
CAR = {"vehicle_in_driveway": True, "plate_visible": True, "what": "a white SUV facing the camera"}
VAN = {"vehicle_in_driveway": True, "plate_visible": False, "what": "a white SUV, plate not readable"}

REPLAYS = {
    "early": {
        "script": [EMPTY],
        "flags": ["--leave-after-s", "6", "--scene-check-after-s", "1"],
        "env": {},
    },
    "held": {
        "script": [{"until": 23, **CAR}, EMPTY],
        "flags": ["--leave-after-s", "4", "--scene-check-after-s", "2", "--scene-relook-every-s", "2"],
        "env": {"SPOTTER_TEST_DROP_READS": "10,20"},
    },
    "unreadable": {
        "script": [{"until": 6, **EMPTY}, {"until": 22, **VAN}, EMPTY],
        "flags": ["--scene-idle-every-s", "2", "--scene-unreadable-after-s", "4"],
        "env": {"SPOTTER_TEST_DROP_READS": "0,60"},
    },
}


def events() -> list[dict]:
    path = REPO_ROOT / "data" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(name: str, plate: str) -> bool:
    spec = REPLAYS[name]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec["script"], handle)
        script = handle.name
    (REPO_ROOT / "data" / "events.jsonl").unlink(missing_ok=True)
    env = {**os.environ, **spec["env"], "SPOTTER_SCENE": f"script:{script}"}
    seed = [PYTHON, "-m", "spotter.seed", "--reset", "--plate", plate, "--driver", "Test Driver", "--hours", "12", "--rate", "5.00"]
    watch = [PYTHON, "-m", "spotter.run", "--source", VIDEO, "--expect", plate, "--zone", CLIP_ZONE,
             "--alerts", "console", "--light", "console", *spec["flags"]]
    print(f"######## SCENE REPLAY {name} ########")
    subprocess.run(seed, cwd=REPO_ROOT, check=False, capture_output=True)
    result = subprocess.run(watch, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    lines = [l for l in result.stdout.splitlines() if not l.startswith("INFO:")]
    for line in lines:
        if line.startswith(("ARRIVED", "LEFT", "UNREADABLE", "SPOT_CLEAR", "SCENE", "scene", "FINISH", "events:", "scene looks")):
            print("  " + line)
    if result.returncode != 0:
        print(result.stderr[-800:])
    seen = events()
    kinds = [e["type"] for e in seen]
    checks = []
    if name == "early":
        left = next((e for e in seen if e["type"] == "LEFT"), None)
        checks = [
            ("one ARRIVED and one LEFT", kinds == ["ARRIVED", "LEFT"]),
            ("LEFT confirmed by scene", bool(left) and left.get("note") == "confirmed by scene"),
            ("LEFT before leave_after_s (6 s) ran out", bool(left) and left["time"] < left["last_seen"] + 6 - 0.5),
            ("left_at is still the last read", bool(left) and abs(left["last_seen"] - 22.22) < 0.6),
        ]
    elif name == "held":
        left = next((e for e in seen if e["type"] == "LEFT"), None)
        checks = [
            ("one ARRIVED and one LEFT", kinds == ["ARRIVED", "LEFT"]),
            ("no LEFT while the plate was covered (10 to 20 s)", bool(left) and left["time"] > 22),
            ("the LEFT came after the real departure, confirmed by the scene", bool(left) and left.get("note") == "confirmed by scene" and left["last_seen"] > 21),
        ]
    elif name == "unreadable":
        checks = [("UNREADABLE_VEHICLE then SPOT_CLEAR, nothing else", kinds == ["UNREADABLE_VEHICLE", "SPOT_CLEAR"])]
        check = subprocess.run([PYTHON, "scripts/check_attendant.py", "unreadable"], cwd=REPO_ROOT, text=True, capture_output=True)
        for line in check.stdout.splitlines():
            if line.startswith(("UNREADABLE", "SPOT_CLEAR", "PASS", "FAIL", "note:")):
                print("  check: " + line)
        checks.append(("attendant check", check.returncode == 0))
    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"  {label}: {'ok' if passed else 'MISSING'}")
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    os.unlink(script)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Replays with scripted scene reports.")
    parser.add_argument("--expect", required=True, metavar="PLATE")
    parser.add_argument("names", nargs="*", help=f"which replays (default all): {', '.join(REPLAYS)}")
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in REPLAYS]
    if unknown:
        parser.error(f"unknown replay {', '.join(unknown)}; choose from {', '.join(REPLAYS)}")
    results = [run(name, args.expect) for name in (args.names or list(REPLAYS))]
    print(f"\n{sum(results)}/{len(results)} scene replays passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
