"""Live sources: a reader that keeps only the newest frame, a clock that counts
only watched time, and a monitor that notices camera outages.

An rtsp:// source is read over TCP with open and read timeouts, so a dead
connection cannot hang the reader, and is reopened with a growing wait (1, 2,
4, 8, then 10 s) whenever it drops or cannot be opened. A video file opened
live is played in real time and treated like a camera; it ends when the file
ends. In both cases a background thread reads continuously and the main loop
takes the newest frame when it wants one, so no frame is ever stale.
"""

import os
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit

import cv2
import numpy as np

GAP_LIMIT_S = 2.0  # a longer gap between checked frames does not count as unread time
OFFLINE_AFTER_S = 10.0  # no new frame for this long: the camera is offline
RETRY_WAITS_S = (1, 2, 4, 8, 10)  # growing wait between reconnects; then 10 s each time
TIMEOUT_S = 10.0  # open and read timeouts for a stream


def is_stream(source: str) -> bool:
    return "://" in source


def mask_source(source: str) -> str:
    """The source as it may be printed: rtsp://user:pass@host/path becomes rtsp://***@host/path."""
    if not is_stream(source):
        return source
    parts = urlsplit(source)
    host = parts.netloc.rsplit("@", 1)[-1]
    prefix = "***@" if "@" in parts.netloc else ""
    return f"{parts.scheme}://{prefix}{host}{parts.path}"


class WatchedClock:
    """Seconds of watched time: time counts only between checked frames less than GAP_LIMIT_S apart.

    A stalled model call, a sleeping laptop or a dropped camera leaves a long
    gap, and a long gap adds nothing, so an outage can never look like a
    departure.
    """

    def __init__(self, gap_limit_s: float = GAP_LIMIT_S) -> None:
        self.gap_limit_s = gap_limit_s
        self.watched = 0.0
        self._last: float | None = None

    def advance(self, real_seconds: float) -> float:
        """Account for a checked frame at `real_seconds`; returns the watched time."""
        if self._last is not None:
            gap = real_seconds - self._last
            if 0 <= gap <= self.gap_limit_s:
                self.watched += gap
        self._last = real_seconds
        return self.watched


class CameraMonitor:
    """Notices when frames stop coming and when they come back, reporting each once."""

    def __init__(self, offline_after_s: float = OFFLINE_AFTER_S) -> None:
        self.offline_after_s = offline_after_s
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.offline_since: float | None = None
        self.outages = 0

    def start(self, now: float) -> None:
        self.started_at = now

    def frame(self, now: float) -> float | None:
        """A new frame arrived. Returns the seconds it was down if this ends an outage."""
        down = None
        if self.offline_since is not None:
            down = now - self.offline_since
            self.offline_since = None
        self.last_frame_at = now
        return down

    def tick(self, now: float) -> bool:
        """No new frame right now. Returns True once when the camera has just gone offline."""
        if self.offline_since is not None:
            return False
        since = self.last_frame_at if self.last_frame_at is not None else self.started_at
        if since is not None and now - since >= self.offline_after_s:
            self.offline_since = since
            self.outages += 1
            return True
        return False


class FrameSource:
    """Reads a source on a background thread and keeps only the newest frame."""

    def __init__(
        self,
        source: str,
        timeout_s: float = TIMEOUT_S,
        hold: Callable[[float], bool] | None = None,
    ) -> None:
        self.source = source
        self.timeout_s = timeout_s
        # Test only: while hold(time.time()) is true no frame is published, and a
        # file stands still, which imitates a camera outage with the car still there.
        self.hold = hold
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_at = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.ended = False  # a file played to its end
        self.opens = 0
        self.failed_opens = 0
        self.width = self.height = 0
        self.fps = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s + 2)

    def newest(self) -> tuple[np.ndarray, float, int] | None:
        """The newest frame, when it was read (time.time()) and its sequence number."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._frame_at, self._seq

    def _open(self) -> cv2.VideoCapture | None:
        if is_stream(self.source):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")  # errors only: no per-packet warnings on the console
            timeout_ms = int(self.timeout_s * 1000)
            capture = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
            )
        else:
            capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            return None
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        return capture

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)

    def _run(self) -> None:
        attempt = 0
        file = not is_stream(self.source)
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self.failed_opens += 1
                self._sleep(RETRY_WAITS_S[min(attempt, len(RETRY_WAITS_S) - 1)])
                attempt += 1
                continue
            attempt = 0
            self.opens += 1
            self.connected = True
            frame_interval = 1 / self.fps if file and self.fps > 0 else 0.0
            next_due = time.monotonic()
            while not self._stop.is_set():
                if self.hold is not None and self.hold(time.time()):
                    self._sleep(0.05)
                    next_due = time.monotonic()
                    continue
                if frame_interval:  # a file plays in real time
                    now = time.monotonic()
                    if now < next_due:
                        self._sleep(next_due - now)
                    next_due = max(next_due + frame_interval, time.monotonic() - frame_interval)
                ok, frame = capture.read()
                if not ok:
                    break
                with self._lock:
                    self._frame, self._frame_at, self._seq = frame, time.time(), self._seq + 1
            capture.release()
            self.connected = False
            if file:  # the end of the file, not a dropped connection
                self.ended = True
                return
            self._sleep(RETRY_WAITS_S[0])
