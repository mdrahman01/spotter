"""Measure Cosmos Reason on real camera frames: is a vehicle in the driveway, is its plate visible?

    .venv/bin/python scripts/scene_eval.py --car FILE... --empty FILE... [--covered-dir data/eval] [--expect PLATE]
    .venv/bin/python scripts/scene_eval.py --eval-frames [data/eval_frames]

With --eval-frames the frames a live run saved once a minute are used, named
..._plate.jpg or ..._noplate.jpg by the plate reader's view at the time, and
the report says how often the scene agreed with the plate reader.

Each file is a camera frame you have labelled by eye: --car frames show the car
parked in the spot, --empty frames show the empty driveway (a car on the street
behind does not count). For every car frame a copy with the plate hidden under
a plain rectangle is made too (the plate is found with the plate reader, so
--expect PLATE is needed), and the expected answer for it is vehicle yes,
plate not visible. Runs look() on all of them with the Cosmos endpoint from
.env and reports, per question, how many were right, the wrong ones by file
name, and the latency (middle and worst). Prints no plate text.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.calibrate import scan_frame  # noqa: E402
from spotter.reader import normalize  # noqa: E402
from spotter.scene import CosmosScene  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def cover_plate(frame, box, pad: float = 0.15):
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    out = frame.copy()
    cv2.rectangle(out, (max(0, x1 - dx), max(0, y1 - dy)), (x2 + dx, y2 + dy), (128, 128, 128), -1)
    return out


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Cosmos accuracy on labelled camera frames.")
    parser.add_argument("--car", nargs="*", default=[], help="frames with the car parked in the spot")
    parser.add_argument("--empty", nargs="*", default=[], help="frames of the empty driveway")
    parser.add_argument("--covered-dir", default="data/eval")
    parser.add_argument("--expect", metavar="PLATE", help="your plate text, to find and cover it")
    parser.add_argument("--results", default="data/eval/results.json")
    parser.add_argument("--eval-frames", nargs="?", const="data/eval_frames", metavar="DIR",
                        help="use the frames a live run saved once a minute (plate / noplate by name)")
    args = parser.parse_args()
    if args.eval_frames:
        frames = sorted(Path(args.eval_frames).glob("*.jpg"))
        args.car = [str(p) for p in frames if p.stem.endswith("_plate")]
        args.empty = [str(p) for p in frames if p.stem.endswith("_noplate")]
        args.expect = None  # no covered copies: the question is only whether the scene agrees with the reader
        print(f"{len(frames)} eval frames: {len(args.car)} with a plate present, {len(args.empty)} without")
    settings = config.load({})
    url, token = os.environ.get("COSMOS_URL", "").strip(), os.environ.get("COSMOS_TOKEN", "").strip()
    if not url or not token:
        print("COSMOS_URL or COSMOS_TOKEN is missing from .env: run scripts/cosmos_up.py first", file=sys.stderr)
        return 2
    scene = CosmosScene(url, token, settings.spot_description, timeout_s=settings.scene_timeout_s, say=print)

    cases = []  # (label, path, expected vehicle, expected plate visible)
    for path in args.car:
        cases.append(("car", path, True, True))
    for path in args.empty:
        cases.append(("empty", path, False, False))
    if args.car and args.expect:
        from fast_alpr import ALPR

        alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr_model="cct-xs-v2-global-model",
                    detector_providers=["CPUExecutionProvider"], ocr_providers=["CPUExecutionProvider"])
        Path(args.covered_dir).mkdir(parents=True, exist_ok=True)
        expected = normalize(args.expect)
        for path in args.car:
            frame = cv2.imread(path)
            boxes = scan_frame(alpr, frame, expected)
            if not boxes:
                print(f"{path}: plate not found by the reader, no covered copy")
                continue
            b = max(boxes, key=lambda b: b.width)
            covered = Path(args.covered_dir) / (Path(path).stem + "_covered.jpg")
            cv2.imwrite(str(covered), cover_plate(frame, (b.x1, b.y1, b.x2, b.y2)))
            cases.append(("covered", str(covered), True, False))

    results = []
    print(f"{len(cases)} frames: {sum(c[0] == 'car' for c in cases)} car, {sum(c[0] == 'empty' for c in cases)} empty, {sum(c[0] == 'covered' for c in cases)} covered")
    for label, path, want_vehicle, want_plate in cases:
        frame = cv2.imread(path)
        started = time.perf_counter()
        report = scene.look(frame)
        latency = time.perf_counter() - started
        if report is None:
            print(f"  {label:8} {Path(path).name:42} UNAVAILABLE ({scene.last_error}) {latency:.1f}s")
            results.append({"label": label, "file": Path(path).name, "unavailable": True, "latency_s": latency})
            continue
        ok_v, ok_p = report.vehicle_in_driveway == want_vehicle, report.plate_visible == want_plate
        print(
            f"  {label:8} {Path(path).name:42} vehicle {str(report.vehicle_in_driveway).lower():5} {'ok ' if ok_v else 'WRONG'}"
            f" | plate {str(report.plate_visible).lower():5} {'ok ' if ok_p else 'WRONG'} | {latency:.1f}s | {report.what!r}"
        )
        results.append({"label": label, "file": Path(path).name, "vehicle": report.vehicle_in_driveway, "plate": report.plate_visible,
                        "want_vehicle": want_vehicle, "want_plate": want_plate, "what": report.what, "latency_s": latency})

    answered = [r for r in results if not r.get("unavailable")]
    latencies = sorted(r["latency_s"] for r in results)
    print("--- results ---")
    if args.eval_frames:
        agreed = [r for r in answered if r["vehicle"] == r["want_vehicle"]]
        print(f"scene agreed with the plate reader on {len(agreed)} of {len(answered)} frames ({100 * len(agreed) / max(1, len(answered)):.1f}%)")
    for question, key, want in (("vehicle_in_driveway", "vehicle", "want_vehicle"), ("plate_visible", "plate", "want_plate")):
        right = [r for r in answered if r[key] == r[want]]
        wrong = [r for r in answered if r[key] != r[want]]
        print(f"{question}: {len(right)} of {len(answered)} right ({100 * len(right) / max(1, len(answered)):.1f}%)"
              + (f"; wrong: {', '.join(r['label'] + ' ' + r['file'] for r in wrong)}" if wrong else ""))
    print(f"unavailable: {len(results) - len(answered)} of {len(results)}")
    if latencies:
        print(f"latency: middle {statistics.median(latencies):.1f}s, worst {latencies[-1]:.1f}s, best {latencies[0]:.1f}s")
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=1))
    print(f"results written to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
