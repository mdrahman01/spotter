"""Watch a video source and turn it into ARRIVED and LEFT events.

    python -m spotter.watch --source samples/IMG_2472.mov [--expect PLATE]
    python -m spotter.watch --source rtsp://CAMERA/stream --leave-after-s 60

The source is a video file or a live stream URL such as rtsp://. Frames are
sampled at process_fps, the plate reader looks inside the zone, and the
arrive/leave rule turns the reads into events. A file is timed by its own
clock (frame index / fps), so it replays as fast as the CPU allows and gives
the same events every time. A live stream is timed by the wall clock.

For each event one JSON line is appended to data/events.jsonl, the frame is
saved under data/snapshots/, and one line is printed. A LEFT event also saves
the last frame in which the plate was read, since the car left at that moment,
not when the rule noticed. With --expect PLATE the
printed lines say MINE or OTHER instead of the plate text. spotter.run adds a
handler that passes each event to the attendant.

The video shows other people's cars: a plate's text is printed or stored only
once it has ARRIVED; every other read is only counted. data/ and samples/ are
gitignored. SPOTTER_* settings can also be put in .env, which keeps a camera
password off the command line.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv

from spotter import config
from spotter.config import Settings
from spotter.events import ArriveLeaveRule, Event, EventFrames
from spotter.privacy import Masker
from spotter.reader import PlateReader, normalize, zone_box

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.jsonl"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# Allowance for floating-point error when comparing a frame's time with the
# next sample time.
EPSILON = 1e-6

EventHandler = Callable[[Event, EventFrames], None]
"""Called with each event and the real times and frames behind it."""

TimerCheck = Callable[[float, datetime, set[str]], list[Event]]
"""Called on every processed frame with the stream time, the real time and the
plates read in the frame; returns timer events (ENDING_SOON, OVERSTAY) to log and
handle like the rule's own."""


class Clock:
    """Event times in seconds: video time for a file, wall-clock time for a live stream."""

    def __init__(self, live: bool, fps: float, name: str, base_time: datetime | None = None) -> None:
        self.live = live
        self.fps = fps
        self.name = name
        # For a file, video time is added to this to get real timestamps: the
        # moment the run started, unless a replay fixes video second 0 elsewhere.
        self.run_started = base_time or datetime.now(timezone.utc)

    def now(self, frame_index: int) -> float:
        return time.time() if self.live else frame_index / self.fps

    def real_time(self, t: float) -> datetime:
        if self.live:
            return datetime.fromtimestamp(t, timezone.utc)
        return self.run_started + timedelta(seconds=t)

    def label(self, t: float) -> str:
        if self.live:
            return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        return f"{t:.2f}s"

    def snapshot_name(self, t: float) -> str:
        if self.live:
            return datetime.fromtimestamp(t).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return f"{self.name}_{t:08.2f}s"


@dataclass
class Summary:
    live: bool
    fps: float
    received: int = 0
    sampled: int = 0
    reads_mine: int = 0
    reads_other: int = 0
    events: Counter = field(default_factory=Counter)
    still_present: list[str] = field(default_factory=list)
    elapsed: float = 0.0


def printable(source: str) -> str:
    """The source as it may be printed: a URL loses any user:password@ and query."""
    if "://" not in source:
        return source
    parts = urlsplit(source)
    return f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}{parts.path}"


def save_snapshot(frame: np.ndarray, name: str) -> str:
    """Save the full frame under data/snapshots/ and return its repo-relative path."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.jpg"
    if not cv2.imwrite(str(path), frame):
        print(f"could not write {path}", file=sys.stderr)
    return str(path.relative_to(REPO_ROOT))


def log_event(event: Event, frames: EventFrames, clock: Clock, masker: Masker) -> None:
    """Append the event to data/events.jsonl and print one line about it."""
    record = {
        "type": event.type,
        "plate": event.plate,
        "time": round(event.time, 3),
        "first_seen": round(event.first_seen, 3),
        "last_seen": round(event.last_seen, 3),
        "reads": event.reads,
        "snapshot": frames.snapshot,
        "last_read_snapshot": frames.last_read_snapshot if event.type == "LEFT" else None,
    }
    DATA_DIR.mkdir(exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as events_file:
        events_file.write(json.dumps(record) + "\n")
    line = (
        f"{event.type:<7} {masker.label(event.plate)} | at {clock.label(event.time)}"
        f" | first seen {clock.label(event.first_seen)}"
        f" | last seen {clock.label(event.last_seen)}"
        f" | {event.reads} reads | {frames.snapshot}"
    )
    if event.type == "LEFT":
        line += f" | last read {frames.last_read_snapshot}"
    print(line, flush=True)


def run(
    settings: Settings,
    masker: Masker,
    handler: EventHandler | None = None,
    timers: TimerCheck | None = None,
    base_time: datetime | None = None,
) -> Summary:
    """Replay a file or follow a live stream to its end, and return the counts.

    Prints the source, the settings and one line per event. Each event also
    goes to the handler, if given, as soon as it happens, with its snapshot
    path and real time. The watcher waits for the handler to return, which for
    a file replay just pauses the video. After the rule's events of a frame,
    `timers` is asked for timer events, given the plates read in the frame, and
    they are logged and handled the same way. `base_time` fixes the real time of
    video second 0 for a file replay.

    Raises FileNotFoundError or RuntimeError if the source cannot be opened.
    """
    source = settings.source
    live = "://" in source
    if not live and not Path(source).is_file():
        raise FileNotFoundError(f"no such video file: {source}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {printable(source)}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not live and fps <= 0:
        raise RuntimeError(f"{source} reports no frame rate")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clock = Clock(live, fps, Path(source).stem, None if live else base_time)
    summary = Summary(live, fps)

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
    if not live and base_time is not None:
        print(f"replay: video second 0 is {clock.run_started.isoformat(timespec='seconds')}")
    if masker.expected:
        print("--expect given: event lines say MINE or OTHER")

    plate_reader = PlateReader(settings.zone, settings.min_det_conf)
    rule = ArriveLeaveRule(
        settings.arrive_reads, settings.arrive_window_s, settings.leave_after_s
    )
    interval = 1 / settings.process_fps
    next_sample = -math.inf
    # The latest frame in which each plate was read, with its stream time: the
    # frame a LEFT event is about, since the car left then, not when the rule fired.
    last_read: dict[str, tuple[float, np.ndarray]] = {}
    started = time.perf_counter()
    try:
        while capture.grab():
            now = clock.now(summary.received)
            summary.received += 1
            if now + EPSILON < next_sample:
                continue
            next_sample = (math.floor(now / interval + EPSILON) + 1) * interval
            ok, frame = capture.retrieve()
            if not ok:
                continue
            summary.sampled += 1
            reads = plate_reader.read(frame)
            mine = sum(read.text == masker.expected for read in reads)
            summary.reads_mine += mine
            summary.reads_other += len(reads) - mine
            plates = {read.text for read in reads}
            for plate in plates:
                last_read[plate] = (now, frame)
            snapshot = None

            def frames_for(event: Event) -> EventFrames:
                nonlocal snapshot
                if snapshot is None:
                    snapshot = save_snapshot(frame, clock.snapshot_name(now))
                if event.type == "LEFT" and event.plate in last_read:
                    read_time, read_frame = last_read.pop(event.plate)
                    read_snapshot = save_snapshot(read_frame, clock.snapshot_name(read_time) + "_last-read")
                    return EventFrames(clock.real_time(now), snapshot, clock.real_time(read_time), read_snapshot)
                return EventFrames(clock.real_time(now), snapshot, clock.real_time(now), snapshot)

            def dispatch(events: list[Event]) -> None:
                for event in events:
                    frames = frames_for(event)
                    log_event(event, frames, clock, masker)
                    summary.events[event.type] += 1
                    if handler is not None:
                        handler(event, frames)

            events = rule.update(now, plates)
            if events:
                dispatch(events)
            if timers is not None:
                # After the rule's events, so a LEFT closes the stay before the timer looks.
                timer_events = timers(now, clock.real_time(now), plates)
                if timer_events:
                    dispatch(timer_events)
            # Frames of plates that are gone and not recently read are no longer needed.
            for plate in [p for p, (t, _) in last_read.items() if p not in rule.present and now - t > settings.leave_after_s]:
                del last_read[plate]
    except KeyboardInterrupt:
        print("stopped by Ctrl+C")
    finally:
        capture.release()
        plate_reader.close()
        last_read.clear()
        gc.collect()
    summary.elapsed = time.perf_counter() - started
    summary.still_present = [masker.label(plate) for plate in sorted(rule.present)]
    return summary


def print_summary(summary: Summary, masker: Masker) -> None:
    print("--- summary ---")
    if summary.live:
        print(
            f"sampled frames: {summary.sampled} of {summary.received} received"
            f" in {summary.elapsed:.0f}s"
        )
    else:
        print(
            f"sampled frames: {summary.sampled} of {summary.received} "
            f"({summary.received / summary.fps:.2f}s of video processed in {summary.elapsed:.1f}s)"
        )
    if masker.expected is None:
        print(f"plate reads: {summary.reads_other} (a text is shown only once its plate arrives)")
    else:
        print(f"reads of my plate: {summary.reads_mine}")
        print(f"reads of other text: {summary.reads_other}")
    counts = ", ".join(f"{count} {kind}" for kind, count in summary.events.items())
    print(f"events: {summary.events.total()}" + (f" ({counts})" if counts else ""))
    print(f"still present at the end: {', '.join(summary.still_present) or 'none'}")


def parse_args(parser: argparse.ArgumentParser) -> tuple[Settings, Masker, argparse.Namespace]:
    """Shared by spotter.watch and spotter.run: settings, the --expect masker, and the
    parsed arguments for any flags the caller added to the parser first."""
    config.add_arguments(parser)
    parser.add_argument(
        "--expect",
        metavar="PLATE",
        help="your plate text: printed output says MINE or OTHER instead of plate "
        "text, and the summary splits the reads",
    )
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    try:
        settings = config.load(vars(args))
    except ValueError as err:
        parser.error(str(err))
    if not settings.source:
        parser.error(f"no source: pass --source or set {config.env_name('source')}")
    expected = None
    if args.expect is not None:
        expected = normalize(args.expect)
        if not expected:
            parser.error("--expect needs at least one letter or digit")
    return settings, Masker(expected), args


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="python -m spotter.watch",
        description="Turn a video file or live stream into ARRIVED and LEFT events.",
    )
    settings, masker, _ = parse_args(parser)
    try:
        summary = run(settings, masker)
    except (FileNotFoundError, RuntimeError) as err:
        print(err, file=sys.stderr)
        return 2
    print_summary(summary, masker)
    return 0


def exit_now(code: int) -> None:
    """Leave with the real exit code once output is flushed.

    The ONNX Runtime and OpenCV libraries sometimes abort inside their own
    teardown at interpreter exit (libc++ "recursive_mutex lock failed"), after
    all our work is done. os._exit skips that teardown.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    exit_now(main())
