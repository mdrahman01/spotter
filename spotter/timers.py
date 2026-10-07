"""Timer events for open stays: ENDING_SOON and OVERSTAY, raised on a sighting.

The watcher owns the clock and, on every processed frame, tells the timer
which plates it read. A timer event fires only on a read of its plate at or
after the event's time, never by the clock alone, so a car last seen before a
deadline gets neither the warning nor the fee. Each event is raised at most
once per stay, which the database remembers; raising OVERSTAY marks the stay
as overstayed at the time of that read. The model never decides that.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from spotter.db import Database, parse

ENDING_SOON = "ENDING_SOON"
OVERSTAY = "OVERSTAY"


@dataclass(frozen=True)
class TimerEvent:
    type: str
    plate: str
    session_id: int
    booking_ends_at: datetime


def time_left_words(seconds: float) -> str:
    """The time left on a booking as the driver is told it, made by code not the model."""
    if seconds < 60:
        return "less than a minute"
    minutes = int(seconds / 60 + 0.5)  # nearest minute, halves up
    return f"about {minutes} minute{'' if minutes == 1 else 's'}"


def due(session: dict, booking: dict, now: datetime, ending_soon_s: float, overstay_grace_s: float) -> list[str]:
    """The timer events an open stay is due for at `now`, in order.

    ENDING_SOON needs time left on the booking: a warning after the end is
    pointless, so none is raised then. OVERSTAY needs the grace to have run out.
    """
    ends_at = parse(booking["ends_at"])
    remaining = (ends_at - now).total_seconds()
    events = []
    if not session["ending_soon_at"] and 0 <= remaining <= ending_soon_s:
        events.append(ENDING_SOON)
    if not session["overstayed_at"] and -remaining >= overstay_grace_s:
        events.append(OVERSTAY)
    return events


class StayTimer:
    def __init__(self, db: Database, ending_soon_s: float, overstay_grace_s: float) -> None:
        self.db = db
        self.ending_soon_s = ending_soon_s
        self.overstay_grace_s = overstay_grace_s

    def check(self, now: datetime, plates_read: Iterable[str]) -> list[TimerEvent]:
        """Raise what is due for every open stay whose plate was read in this frame, once each."""
        read = set(plates_read)
        events = []
        for session in self.db.open_sessions():
            if session["plate"] not in read:
                continue
            booking = self.db.booking(session["booking_id"])
            for kind in due(session, booking, now, self.ending_soon_s, self.overstay_grace_s):
                if kind == ENDING_SOON:
                    self.db.mark_ending_soon(session["id"], now)
                else:
                    self.db.mark_overstayed(session["id"], now)
                events.append(TimerEvent(kind, session["plate"], session["id"], parse(booking["ends_at"])))
        return events
