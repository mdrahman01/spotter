"""Watch a video source and turn it into ARRIVED and LEFT events.

    python -m spotter.watch --source samples/IMG_2472.mov [--expect PLATE]
    python -m spotter.watch --source rtsp://CAMERA/stream --leave-after-s 60

The source is a video file or a live stream URL such as rtsp://. Frames are
sampled at process_fps, the plate reader looks inside the zone, and the
arrive/leave rule turns the reads into events. A file is timed by its own
clock (frame index / fps), so it replays as fast as the CPU allows and gives
the same events every time. A live stream is timed by the wall clock.

For each event one JSON line is appended to data/events.jsonl, the frame is
saved under data/snapshots/, and one line is printed. With --expect PLATE the
printed lines say MINE or OTHER instead of the plate text.

The video shows other people's cars: a plate's text is printed or stored only
once it has ARRIVED; every other read is only counted. data/ and samples/ are
gitignored. SPOTTER_* settings can also be put in .env, which keeps a camera
password off the command line.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv

from spotter import config
from spotter.events import ArriveLeaveRule, Event
from spotter.reader import PlateReader, normalize, zone_box

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.jsonl"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# Allowance for floating-point error when comparing a frame's time with the
# next sample time.
EPSILON = 1e-6


class Clock:
    """Event times in seconds: video time for a file, wall-clock time for a live stream."""

    def __init__(self, live: bool, fps: float, name: str) -> None:
        self.live = live
        self.fps = fps
        self.name = name

    def now(self, frame_index: int) -> float:
        return time.time() if self.live else frame_index / self.fps

    def label(self, t: float) -> str:
        if self.live:
            return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        return f"{t:.2f}s"

    def snapshot_name(self, t: float) -> str:
        if self.live:
            return datetime.fromtimestamp(t).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return f"{self.name}_{t:08.2f}s"


def printable(source: str) -> str:
    """The source as it may be printed: a URL loses any user:password@ and query."""
    if "://" not in source:
        return source
    parts = urlsplit(source)
    return f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}{parts.path}"


def who(plate: str, expected: str | None) -> str:
    """How an arrived plate is printed: MINE or OTHER with --expect, else its text."""
    if expected is None:
        return plate
    return "MINE" if plate == expected else "OTHER"


def save_snapshot(frame: np.ndarray, name: str) -> str:
    """Save the full frame under data/snapshots/ and return its repo-relative path."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.jpg"
    if not cv2.imwrite(str(path), frame):
        print(f"could not write {path}", file=sys.stderr)
    return str(path.relative_to(REPO_ROOT))


def log_event(event: Event, snapshot: str, clock: Clock, expected: str | None) -> None:
    """Append the event to data/events.jsonl and print one line about it."""
    record = {
        "type": event.type,
        "plate": event.plate,
        "time": round(event.time, 3),
        "first_seen": round(event.first_seen, 3),
        "last_seen": round(event.last_seen, 3),
        "reads": event.reads,
        "snapshot": snapshot,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as events_file:
        events_file.write(json.dumps(record) + "\n")
    print(
        f"{event.type:<7} {who(event.plate, expected)} | at {clock.label(event.time)}"
        f" | first seen {clock.label(event.first_seen)}"
        f" | last seen {clock.label(event.last_seen)}"
        f" | {event.reads} reads | {snapshot}"
    )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="python -m spotter.watch",
        description="Turn a video file or live stream into ARRIVED and LEFT events.",
    )
    config.add_arguments(parser)
    parser.add_argument(
        "--expect",
        metavar="PLATE",
        help="your plate text: event lines say MINE or OTHER instead of plate "
        "text, and the summary splits the reads",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    try:
        settings = config.load(vars(args))
    except ValueError as err:
        parser.error(str(err))
    source = settings.source
    if not source:
        parser.error(f"no source: pass --source or set {config.env_name('source')}")
    expected = None
    if args.expect is not None:
        expected = normalize(args.expect)
        if not expected:
            parser.error("--expect needs at least one letter or digit")

    live = "://" in source
    if not live and not Path(source).is_file():
        print(f"no such video file: {source}", file=sys.stderr)
        return 2
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"could not open {printable(source)}", file=sys.stderr)
        return 2
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not live and fps <= 0:
        print(f"{source} reports no frame rate", file=sys.stderr)
        return 2
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clock = Clock(live, fps, Path(source).stem)

    if live:
        print(f"source: {printable(source)} (live stream, timed by the wall clock)")
    else:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(
            f"source: {source} (file, {width}x{height}, {fps:.2f} fps, "
            f"{frames:.0f} frames, {frames / fps:.2f}s, timed by video time)"
        )
    x0, y0, x1, y1 = zone_box(settings.zone, width, height)
    print(
        f"zone {','.join(str(v) for v in settings.zone)} = x {x0}-{x1}, y {y0}-{y1} px"
        f" | process_fps {settings.process_fps} | min_det_conf {settings.min_det_conf}"
        f" | arrive: {settings.arrive_reads} reads within {settings.arrive_window_s}s"
        f" | leave: {settings.leave_after_s}s unread"
    )
    if expected:
        print("--expect given: event lines say MINE or OTHER")

    plate_reader = PlateReader(settings.zone, settings.min_det_conf)
    rule = ArriveLeaveRule(
        settings.arrive_reads, settings.arrive_window_s, settings.leave_after_s
    )
    interval = 1 / settings.process_fps
    next_sample = -math.inf
    frame_index = sampled = reads_mine = reads_other = 0
    event_counts: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        while capture.grab():
            now = clock.now(frame_index)
            frame_index += 1
            if now + EPSILON < next_sample:
                continue
            next_sample = (math.floor(now / interval + EPSILON) + 1) * interval
            ok, frame = capture.retrieve()
            if not ok:
                continue
            sampled += 1
            reads = plate_reader.read(frame)
            mine = sum(read.text == expected for read in reads)
            reads_mine += mine
            reads_other += len(reads) - mine
            events = rule.update(now, [read.text for read in reads])
            if events:
                snapshot = save_snapshot(frame, clock.snapshot_name(now))
                for event in events:
                    log_event(event, snapshot, clock, expected)
                    event_counts[event.type] += 1
    except KeyboardInterrupt:
        print("stopped by Ctrl+C")
    finally:
        capture.release()
    elapsed = time.perf_counter() - started

    print("--- summary ---")
    if live:
        print(f"sampled frames: {sampled} of {frame_index} received in {elapsed:.0f}s")
    else:
        print(
            f"sampled frames: {sampled} of {frame_index} "
            f"({frame_index / fps:.2f}s of video processed in {elapsed:.1f}s)"
        )
    if expected is None:
        print(f"plate reads: {reads_other} (a text is shown only once its plate arrives)")
    else:
        print(f"reads of my plate: {reads_mine}")
        print(f"reads of other text: {reads_other}")
    print(
        f"events: {event_counts.total()} "
        f"({event_counts['ARRIVED']} ARRIVED, {event_counts['LEFT']} LEFT)"
    )
    still_present = [who(plate, expected) for plate in sorted(rule.present)]
    print(f"still present at the end: {', '.join(still_present) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
