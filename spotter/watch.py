"""Watch a video source and turn it into ARRIVED and LEFT events.

    python -m spotter.watch --source samples/IMG_2472.mov [--expect PLATE]          # replay, as fast as the CPU allows
    python -m spotter.watch --source samples/IMG_2472.mov --live [--expect PLATE]   # the file played in real time, like a camera
    python -m spotter.watch --source rtsp://CAMERA/stream --leave-after-s 60        # the camera

Replay: frames are sampled at process_fps and timed by the video's own clock, so
a file gives the same events every run. Live: a background reader keeps only the
newest frame and the loop takes it process_fps times a second, so no frame is
stale. Live time counts toward "unread" only between checked frames less than
2 s apart, so a stalled model call, a sleeping laptop or a dropped camera never
looks like a departure. No new frame for 10 s is a camera outage: CAMERA_OFFLINE
is recorded (spotter.run tells the owner, from code), stays and the light stay as
they are, the connection is retried with a growing wait, and CAMERA_BACK says
how long it was down. The same happens when the source cannot be opened at
start. Ctrl-C finishes the event in hand, prints the summary and exits 0; on
macOS a live run keeps the machine awake.

For each event one JSON line is appended to data/events.jsonl, the frame is
saved under data/snapshots/, and one line is printed. A LEFT event also saves
the last frame in which the plate was read, since the car left at that moment,
not when the rule noticed. With --expect PLATE the printed lines say MINE or
OTHER instead of the plate text. A stream address is printed as
rtsp://***@host/path. spotter.run adds a handler that passes each event to the
attendant.

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
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from spotter import config
from spotter.config import Settings
from spotter.events import ArriveLeaveRule, Event, EventFrames
from spotter.live import CameraMonitor, FrameSource, WatchedClock, is_stream, mask_source
from spotter.privacy import Masker
from spotter.reader import PlateReader, normalize, zone_box
from spotter.words import clock_words, duration_words

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.jsonl"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# Allowance for floating-point error when comparing a frame's time with the
# next sample time.
EPSILON = 1e-6

CAMERA_OFFLINE, CAMERA_BACK = "CAMERA_OFFLINE", "CAMERA_BACK"

EventHandler = Callable[[Event, EventFrames], None]
"""Called with each event and the real times and frames behind it."""

TimerCheck = Callable[[float, datetime, set[str]], list[Event]]
"""Called on every checked frame with the rule's time, the real time and the plates
read in the frame; returns timer events (ENDING_SOON, OVERSTAY) to log and handle
like the rule's own."""

CameraHandler = Callable[[str, datetime, float | None], None]
"""Called with CAMERA_OFFLINE (downtime None) or CAMERA_BACK (seconds down)."""


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
    watched: float = 0.0  # live: seconds that counted toward "unread"
    longest_gap: float | None = None  # longest gap between reads of a present plate, in rule time
    outages: int = 0
    downtime: float = 0.0
    stopped_by_user: bool = False


class StopRequest:
    """Set by the first Ctrl-C; the run ends after the event in hand. A second Ctrl-C forces out."""

    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        def handler(signum, frame):
            if self.requested:
                raise KeyboardInterrupt
            self.requested = True
            print("Ctrl-C: stopping after the event in hand (press again to force)", flush=True)

        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:  # not the main thread
            pass

    @staticmethod
    def restore() -> None:
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:
            pass


def save_snapshot(frame: np.ndarray, name: str) -> str:
    """Save the full frame under data/snapshots/ and return its repo-relative path."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.jpg"
    if not cv2.imwrite(str(path), frame):
        print(f"could not write {path}", file=sys.stderr)
    return str(path.relative_to(REPO_ROOT))


def append_record(record: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as events_file:
        events_file.write(json.dumps(record) + "\n")


def keep_awake() -> subprocess.Popen | None:
    """On macOS, stop the machine idle-sleeping while this process runs."""
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL)


def test_hold() -> Callable[[float], bool] | None:
    """Test only: SPOTTER_TEST_OUTAGE=start,duration withholds frames for that window of a live run."""
    spec = os.environ.get("SPOTTER_TEST_OUTAGE", "").strip()
    if not spec:
        return None
    start, duration = (float(part) for part in spec.split(","))
    begins = time.time() + start
    print(f"TEST: frames will be withheld from {start:g}s to {start + duration:g}s into the run", flush=True)
    return lambda now: begins <= now < begins + duration


def run(
    settings: Settings,
    masker: Masker,
    handler: EventHandler | None = None,
    timers: TimerCheck | None = None,
    base_time: datetime | None = None,
    live: bool = False,
    on_camera: CameraHandler | None = None,
) -> Summary:
    """Replay a file, or follow a camera (or a file played live) until it ends or Ctrl-C.

    Prints the source, the settings and one line per event. Each event also
    goes to the handler, if given, as soon as it happens, with its snapshot
    paths and real times; the watcher waits for the handler to return. After the
    rule's events of a frame, `timers` is asked for timer events, given the
    plates read in the frame, and they are logged and handled the same way.
    `base_time` fixes the real time of video second 0 for a replay. Camera
    outages go to `on_camera`.

    Raises FileNotFoundError or RuntimeError if a file cannot be opened for replay.
    """
    source = settings.source
    live = live or is_stream(source)
    if not live and not Path(source).is_file():
        raise FileNotFoundError(f"no such video file: {source}")
    summary = Summary(live, 0.0)
    plate_reader = PlateReader(settings.zone, settings.min_det_conf)
    rule = ArriveLeaveRule(settings.arrive_reads, settings.arrive_window_s, settings.leave_after_s)
    print(
        f"zone {','.join(str(v) for v in settings.zone)}"
        f" | process_fps {settings.process_fps} | min_det_conf {settings.min_det_conf}"
        f" | arrive: {settings.arrive_reads} reads within {settings.arrive_window_s}s"
        f" | leave: {settings.leave_after_s}s unread"
    )
    if masker.expected:
        print("--expect given: event lines say MINE or OTHER")

    # The latest frame in which each plate was read, with the rule's time and the
    # real time: the frame a LEFT event is about. And when each present plate was
    # first read, for how long it stayed.
    last_read: dict[str, tuple[float, datetime, np.ndarray]] = {}
    first_read: dict[str, datetime] = {}
    stop = StopRequest()

    def label(t: float, real_at: datetime) -> str:
        return clock_words(real_at) if live else f"{t:.2f}s"

    def snapshot_name(t: float, real_at: datetime) -> str:
        if live:
            return real_at.astimezone().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return f"{Path(source).stem}_{t:08.2f}s"

    def log_event(event: Event, frames: EventFrames) -> None:
        append_record(
            {
                "type": event.type,
                "plate": event.plate,
                "time": round(event.time, 3),
                "first_seen": round(event.first_seen, 3),
                "last_seen": round(event.last_seen, 3),
                "reads": event.reads,
                "at": frames.at.isoformat(timespec="seconds"),
                "last_read_at": frames.last_read_at.isoformat(timespec="seconds"),
                "snapshot": frames.snapshot,
                "last_read_snapshot": frames.last_read_snapshot if event.type == "LEFT" else None,
            }
        )
        line = (
            f"{event.type:<7} {masker.label(event.plate)} | at {label(event.time, frames.at)}"
            f" | first seen {label(event.first_seen, frames.first_read_at or frames.at)}"
            f" | last seen {label(event.last_seen, frames.last_read_at)}"
            f" | {event.reads} reads | {frames.snapshot}"
        )
        if event.type == "LEFT":
            line += f" | last read {frames.last_read_snapshot}"
        print(line, flush=True)

    def camera_event(kind: str, when: datetime, down: float | None) -> None:
        summary.events[kind] += 1
        if kind == CAMERA_BACK and down is not None:
            summary.downtime += down
        else:
            summary.outages += 1
        detail = f"no new frame for {duration_words(settings_offline_s)}" if kind == CAMERA_OFFLINE else f"down {duration_words(down or 0)}"
        print(f"{kind} | at {clock_words(when)} | {detail}", flush=True)
        append_record({"type": kind, "at": when.isoformat(timespec="seconds"), "downtime_s": round(down, 1) if down else None})
        if on_camera is not None:
            on_camera(kind, when, down)

    settings_offline_s = CameraMonitor().offline_after_s

    def process(frame: np.ndarray, now: float, real_at: datetime) -> None:
        """One checked frame: read plates, update the rule and the timers, dispatch events."""
        summary.sampled += 1
        reads = plate_reader.read(frame)
        mine = sum(read.text == masker.expected for read in reads)
        summary.reads_mine += mine
        summary.reads_other += len(reads) - mine
        plates = {read.text for read in reads}
        for plate in plates:
            if plate in rule.present and plate in last_read:
                gap = now - last_read[plate][0]
                summary.longest_gap = gap if summary.longest_gap is None else max(summary.longest_gap, gap)
            last_read[plate] = (now, real_at, frame)
        snapshot = None

        def frames_for(event: Event) -> EventFrames:
            nonlocal snapshot
            if snapshot is None:
                snapshot = save_snapshot(frame, snapshot_name(now, real_at))
            if event.type == "ARRIVED":
                first_read[event.plate] = real_at - timedelta(seconds=max(0.0, event.time - event.first_seen))
            if event.type == "LEFT" and event.plate in last_read:
                read_t, read_at, read_frame = last_read.pop(event.plate)
                read_snapshot = save_snapshot(read_frame, snapshot_name(read_t, read_at) + "_last-read")
                return EventFrames(real_at, snapshot, read_at, read_snapshot, first_read.pop(event.plate, None))
            return EventFrames(real_at, snapshot, real_at, snapshot, first_read.get(event.plate))

        def dispatch(events: list[Event]) -> None:
            for event in events:
                frames = frames_for(event)
                log_event(event, frames)
                summary.events[event.type] += 1
                if handler is not None:
                    handler(event, frames)

        events = rule.update(now, plates)
        if events:
            dispatch(events)
        if timers is not None:
            # After the rule's events, so a LEFT closes the stay before the timer looks.
            timer_events = timers(now, real_at, plates)
            if timer_events:
                dispatch(timer_events)
        # Frames of plates that are gone and not recently read are no longer needed.
        for plate in [p for p, (t, _, _) in last_read.items() if p not in rule.present and now - t > settings.leave_after_s]:
            del last_read[plate]

    started = time.perf_counter()
    stop.install()
    awake = keep_awake() if live else None
    try:
        if live:
            run_live(settings, summary, process, camera_event, stop)
        else:
            run_replay(settings, summary, process, base_time, stop)
    except KeyboardInterrupt:
        print("stopped by Ctrl-C", flush=True)
        summary.stopped_by_user = True
    finally:
        StopRequest.restore()
        if awake is not None:
            awake.terminate()
        plate_reader.close()
        last_read.clear()
        gc.collect()
    summary.elapsed = time.perf_counter() - started
    summary.stopped_by_user = summary.stopped_by_user or stop.requested
    summary.still_present = [masker.label(plate) for plate in sorted(rule.present)]
    return summary


def run_replay(settings: Settings, summary: Summary, process, base_time: datetime | None, stop: StopRequest) -> None:
    """Every sampled frame of the file, timed by the video's own clock."""
    source = settings.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"{source} reports no frame rate")
    summary.fps = fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    run_started = base_time or datetime.now(timezone.utc)  # video second 0
    print(
        f"source: {source} (file, {width}x{height}, {fps:.2f} fps, {frames:.0f} frames,"
        f" {frames / fps:.2f}s, timed by video time)"
    )
    x0, y0, x1, y1 = zone_box(settings.zone, width, height)
    print(f"zone in pixels: x {x0}-{x1}, y {y0}-{y1}")
    if base_time is not None:
        print(f"replay: video second 0 is {run_started.isoformat(timespec='seconds')}")
    interval = 1 / settings.process_fps
    next_sample = -math.inf
    try:
        while capture.grab() and not stop.requested:
            now = summary.received / fps
            summary.received += 1
            if now + EPSILON < next_sample:
                continue
            next_sample = (math.floor(now / interval + EPSILON) + 1) * interval
            ok, frame = capture.retrieve()
            if ok:
                process(frame, now, run_started + timedelta(seconds=now))
    finally:
        capture.release()


def run_live(settings: Settings, summary: Summary, process, camera_event, stop: StopRequest) -> None:
    """The newest frame, process_fps times a second, until the source ends or Ctrl-C."""
    source = settings.source
    kind = "camera" if is_stream(source) else "file played live"
    print(f"source: {mask_source(source)} ({kind}; waiting for the first frame)")
    frames = FrameSource(source, hold=test_hold())
    monitor = CameraMonitor()
    clock = WatchedClock()
    frames.start()
    monitor.start(time.time())
    interval = 1 / settings.process_fps
    next_tick = time.monotonic()
    last_seq = None
    announced = False
    try:
        while not stop.requested:
            wait = next_tick - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.05))
                continue
            next_tick += interval
            newest = frames.newest()
            if newest is None or newest[2] == last_seq:
                if frames.ended:
                    break
                if monitor.tick(time.time()):
                    camera_event(CAMERA_OFFLINE, datetime.now(timezone.utc), None)
                continue
            frame, captured_at, seq = newest
            last_seq = seq
            summary.received = seq
            if not announced:
                summary.fps = frames.fps
                x0, y0, x1, y1 = zone_box(settings.zone, frames.width, frames.height)
                print(f"first frame: {frames.width}x{frames.height} | zone in pixels: x {x0}-{x1}, y {y0}-{y1}", flush=True)
                announced = True
            down = monitor.frame(captured_at)
            if down is not None:
                camera_event(CAMERA_BACK, datetime.fromtimestamp(captured_at, timezone.utc), down)
            now = clock.advance(captured_at)
            process(frame, now, datetime.fromtimestamp(captured_at, timezone.utc))
            summary.watched = clock.watched
            next_tick = max(next_tick, time.monotonic())  # a slow event does not cause a burst of catch-up ticks
    finally:
        frames.stop()


def print_summary(summary: Summary, masker: Masker) -> None:
    print("--- summary ---")
    if summary.live:
        print(
            f"checked frames: {summary.sampled} (newest of {summary.received} received)"
            f" in {summary.elapsed:.0f}s; watched time {summary.watched:.1f}s"
        )
        print(f"camera outages: {summary.outages}, down {duration_words(summary.downtime)} in all")
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
    gap = "none (no plate was read twice while present)" if summary.longest_gap is None else f"{summary.longest_gap:.1f}s"
    print(f"longest gap between reads of a present plate: {gap} (leave_after_s must stay above this)")
    print(f"still present at the end: {', '.join(summary.still_present) or 'none'}")
    if summary.stopped_by_user:
        print("stopped by Ctrl-C")


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
    parser.add_argument(
        "--live",
        action="store_true",
        help="play a video file in real time and treat it like a camera (rtsp:// sources are always live)",
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
    settings, masker, args = parse_args(parser)
    try:
        summary = run(settings, masker, live=args.live)
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
