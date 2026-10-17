"""The arrive/leave rule, free of video and model code so it can be tested alone.

A plate ARRIVES when the same text is read arrive_reads times within
arrive_window_s seconds. It LEAVES once it has not been read for
leave_after_s seconds. Times are seconds on the caller's clock (video time
for a file, wall-clock time for a live stream) and must never go backwards.

One car, one plate: while a plate is present, a read within two edits of it
is the same car misread and counts as a sighting of the present plate, never
as a new car; a text three edits away is the same car too unless it was read
in the same frame as the present plate at least once in its window (two cars
show two plates at once; one plate misread never shows both). Once judged a
misread, a text keeps counting as that plate while the plate is present. At
arrival the text that reached the rule is compared with the near-duplicates
read in the same window and the text read most often wins.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

# Allowance for floating-point error when comparing times, e.g. 26.0 - 22.0 with 4.
EPSILON = 1e-6
MISREAD_EDITS = 2  # a read this close to a present plate is that plate, misread
FAR_MISREAD_EDITS = 3  # ... and this close, if it was never read beside the present plate


def edit_distance(a: str, b: str, limit: int = MISREAD_EDITS) -> int:
    """Levenshtein distance, capped at limit + 1 (plates are short, so this is cheap)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Event:
    type: str  # "ARRIVED", "LEFT", a timer event, or a scene event (UNREADABLE_VEHICLE, SPOT_CLEAR)
    plate: str  # "" for a scene event about a vehicle with no readable plate
    time: float  # when the rule fired
    first_seen: float  # first read of this stay
    last_seen: float  # latest read of this stay
    reads: int  # reads of the plate during this stay
    note: str = ""  # e.g. "confirmed by scene"
    what: str = ""  # scene events: the vehicle as the scene described it


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
        # Plates whose departure is held off because the scene still sees the car.
        self.held: set[str] = set()
        # Read times of plates that have not arrived, newest last.
        self._recent: dict[str, deque[float]] = {}
        # Frames in which a not-yet-arrived text was read together with a present plate.
        self._with_present: dict[str, int] = {}
        # Texts judged to be misreads of a present plate, and that plate.
        self._aliases: dict[str, str] = {}
        self.merged = 0  # reads of a near-duplicate counted as the plate they were misread from

    def same_car(self, text: str) -> str | None:
        """The present plate this read is a misread of (within two edits, or judged one before), if any."""
        if text in self.present:
            return text
        alias = self._aliases.get(text)
        if alias in self.present:
            return alias
        nearest = min(
            ((edit_distance(text, present), present) for present in self.present),
            default=(MISREAD_EDITS + 1, None),
        )
        return nearest[1] if nearest[0] <= MISREAD_EDITS else None

    def resolve(self, plates: Iterable[str], count: bool = True) -> set[str]:
        """The plates of a frame with misreads of a present plate replaced by that plate."""
        resolved = set()
        for text in plates:
            present = self.same_car(text)
            if present is not None and present != text and count:
                self.merged += 1
            resolved.add(present if present is not None else text)
        return resolved

    def hold(self, plate: str) -> None:
        """Keep the plate's stay open past leave_after_s until release() or another plate arrives."""
        if plate in self.present:
            self.held.add(plate)

    def release(self, plate: str) -> None:
        self.held.discard(plate)

    def leave_now(self, plate: str, now: float, note: str = "") -> Event | None:
        """Declare the plate gone now, as the scene confirmed; its last read stays the departure time."""
        stay = self.present.pop(plate, None)
        self.held.discard(plate)
        self._forget_aliases(plate)
        if stay is None:
            return None
        return _event("LEFT", plate, now, stay, note)

    def update(self, now: float, plates: Iterable[str]) -> list[Event]:
        """Feed the plates read in one sampled frame at time `now`; return the events.

        Call it for every sampled frame, including frames with no reads, so
        that departures are noticed. A plate read twice in one frame counts once.
        """
        events = []
        resolved = self.resolve(plates)  # the rule counts what it merges
        present_read = any(p in self.present for p in resolved)
        for plate in sorted(resolved):
            stay = self.present.get(plate)
            if stay:
                stay.last_seen = now
                stay.reads += 1
                continue
            times = self._recent.setdefault(plate, deque())
            times.append(now)
            while now - times[0] > self.arrive_window_s + EPSILON:
                times.popleft()
            if present_read:
                self._with_present[plate] = self._with_present.get(plate, 0) + 1
            if len(times) >= self.arrive_reads:
                nearest = min(self.present, key=lambda p: edit_distance(plate, p, limit=FAR_MISREAD_EDITS), default=None)
                if (nearest is not None and not self._with_present.get(plate)
                        and edit_distance(plate, nearest, limit=FAR_MISREAD_EDITS) <= FAR_MISREAD_EDITS):
                    # Close to a present plate and never seen beside it: that car misread, not a second car.
                    self._misread_of_present(plate, nearest, now, len(times))
                    continue
                plate, times = self._most_read_variant(plate, now)
                # A new arrival means a held car has gone: it leaves first.
                for held in sorted(self.held):
                    gone = self.leave_now(held, now, "another plate arrived")
                    if gone:
                        events.append(gone)
                stay = Stay(first_seen=times[0], last_seen=now, reads=len(times))
                self.present[plate] = stay
                events.append(_event("ARRIVED", plate, now, stay))

        for plate in sorted(self.present):
            stay = self.present[plate]
            if plate in self.held:
                continue
            if now - stay.last_seen >= self.leave_after_s - EPSILON:
                del self.present[plate]
                self._forget_aliases(plate)
                events.append(_event("LEFT", plate, now, stay))

        # Forget reads that can no longer be part of any arrival window.
        stale = [p for p, t in self._recent.items() if now - t[-1] > self.arrive_window_s + EPSILON]
        for plate in stale:
            del self._recent[plate]
            self._with_present.pop(plate, None)
        return events

    def _misread_of_present(self, text: str, plate: str, now: float, reads: int) -> None:
        """Count the text's reads as sightings of `plate`, and keep treating the text as that plate."""
        stay = self.present[plate]
        stay.last_seen = now
        stay.reads += reads
        self.merged += reads
        self._aliases[text] = plate
        del self._recent[text]
        self._with_present.pop(text, None)

    def _forget_aliases(self, plate: str) -> None:
        for text in [t for t, p in self._aliases.items() if p == plate]:
            del self._aliases[text]


    def _most_read_variant(self, plate: str, now: float) -> tuple[str, deque[float]]:
        """At arrival: among the texts within two edits of `plate` read in the window, the one read
        most often (ties go to the text that reached the rule); their reads are pooled."""
        window = self.arrive_window_s + EPSILON
        variants = {
            text: deque(t for t in times if now - t <= window)
            for text, times in self._recent.items()
            if text == plate or edit_distance(text, plate) <= MISREAD_EDITS
        }
        chosen = max(variants, key=lambda text: (len(variants[text]), text == plate))
        pooled = sorted(t for times in variants.values() for t in times)
        self.merged += sum(len(times) for text, times in variants.items() if text != chosen)
        for text in variants:
            del self._recent[text]
            self._with_present.pop(text, None)
        return chosen, deque(pooled)


def _event(kind: str, plate: str, now: float, stay: Stay, note: str = "") -> Event:
    return Event(kind, plate, now, stay.first_seen, stay.last_seen, stay.reads, note)
