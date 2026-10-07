"""Timer events for open stays: ENDING_SOON and OVERSTAY.

The watcher owns the clock, so it asks the timer on every processed frame.
Each event is raised at most once per stay, which the database remembers:
ENDING_SOON when the booking ends within ending_soon_s, OVERSTAY when the car
is still present overstay_grace_s after the booking's end. Raising OVERSTAY
marks the stay as overstayed; the model never decides that.
"""

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

    def check(self, now: datetime) -> list[TimerEvent]:
        """Raise whatever is due for every open stay, and record it so it is raised once."""
        events = []
        for session in self.db.open_sessions():
            booking = self.db.booking(session["booking_id"])
            for kind in due(session, booking, now, self.ending_soon_s, self.overstay_grace_s):
                if kind == ENDING_SOON:
                    self.db.mark_ending_soon(session["id"], now)
                else:
                    self.db.mark_overstayed(session["id"], now)
                events.append(TimerEvent(kind, session["plate"], session["id"], parse(booking["ends_at"])))
        return events
