"""The arrive/leave rule, free of video and model code so it can be tested alone.

A plate ARRIVES when the same text is read arrive_reads times within
arrive_window_s seconds. It LEAVES once it has not been read for
leave_after_s seconds. Times are seconds on the caller's clock (video time
for a file, wall-clock time for a live stream) and must never go backwards.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

# Allowance for floating-point error when comparing times, e.g. 26.0 - 22.0 with 4.
EPSILON = 1e-6


@dataclass(frozen=True)
class Event:
    type: str  # "ARRIVED" or "LEFT"
    plate: str
    time: float  # when the rule fired
    first_seen: float  # first read of this stay
    last_seen: float  # latest read of this stay
    reads: int  # reads of the plate during this stay


@dataclass(frozen=True)
class EventFrames:
    """What the watcher hands over with an event: the real times and frames behind it.

    For LEFT the plate was last read some seconds before the rule fired, so
    last_read_at and last_read_snapshot differ from at and snapshot. For every
    other event they are the same moment and the same frame.
    """

    at: datetime  # real time the event fired
    snapshot: str  # repo-relative path of the frame when it fired
    last_read_at: datetime  # real time the plate was last read
    last_read_snapshot: str  # repo-relative path of that frame
    first_read_at: datetime | None = None  # real time the plate was first read in this stay


@dataclass
class Stay:
    first_seen: float
    last_seen: float
    reads: int


class ArriveLeaveRule:
    def __init__(self, arrive_reads: int, arrive_window_s: float, leave_after_s: float) -> None:
        self.arrive_reads = arrive_reads
        self.arrive_window_s = arrive_window_s
        self.leave_after_s = leave_after_s
        self.present: dict[str, Stay] = {}
        # Read times of plates that have not arrived, newest last.
        self._recent: dict[str, deque[float]] = {}

    def update(self, now: float, plates: Iterable[str]) -> list[Event]:
        """Feed the plates read in one sampled frame at time `now`; return the events.

        Call it for every sampled frame, including frames with no reads, so
        that departures are noticed. A plate read twice in one frame counts once.
        """
        events = []
        for plate in sorted(set(plates)):
            stay = self.present.get(plate)
            if stay:
                stay.last_seen = now
                stay.reads += 1
                continue
            times = self._recent.setdefault(plate, deque())
            times.append(now)
            while now - times[0] > self.arrive_window_s + EPSILON:
                times.popleft()
            if len(times) >= self.arrive_reads:
                del self._recent[plate]
                stay = Stay(first_seen=times[0], last_seen=now, reads=len(times))
                self.present[plate] = stay
                events.append(_event("ARRIVED", plate, now, stay))

        for plate in sorted(self.present):
            stay = self.present[plate]
            if now - stay.last_seen >= self.leave_after_s - EPSILON:
                del self.present[plate]
                events.append(_event("LEFT", plate, now, stay))

        # Forget reads that can no longer be part of any arrival window.
        stale = [p for p, t in self._recent.items() if now - t[-1] > self.arrive_window_s + EPSILON]
        for plate in stale:
            del self._recent[plate]
        return events


def _event(kind: str, plate: str, now: float, stay: Stay) -> Event:
    return Event(kind, plate, now, stay.first_seen, stay.last_seen, stay.reads)
