"""The only things the model can do: six plain functions, each with an OpenAI tool schema.

The code, not the model, enforces the rules and does the arithmetic:
- a tool acts only on the plate of the current event;
- a plate has one open stay at a time, and a stay needs a booking covering its start;
- the bill is per started minute at the booking's rate, with a one-minute
  minimum, rounded up to whole cents, and never below the minimum charge; an
  overstayed stay pays a flat overstay fee on top.
A broken rule comes back to the model as {"error": ...}; nothing raises. Error
messages never contain plate text, so they can be printed as they are. Money
is shown to the model as dollars ("$5.00/hour"), never as cents, and every
clock time or duration as words made by code ("6:02 pm", "16 seconds"), which
the WordBook remembers for the finish check.
"""

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime

from spotter.alerts import Alert, Alerts
from spotter.db import Database, parse
from spotter.light import COLORS, Light
from spotter.reader import normalize
from spotter.words import WordBook

PLATE_RULE = "this tool may only act on the plate in the current event"
DEFAULT_MIN_CHARGE_CENTS = 100
DEFAULT_OVERSTAY_FEE_CENTS = 500


@dataclass(frozen=True)
class EventContext:
    """The event the tools act on."""

    type: str  # ARRIVED, ENDING_SOON, OVERSTAY or LEFT
    plate: str
    at: datetime  # the real time the event fired
    snapshot: str  # repo-relative path of the frame when it fired
    last_read_at: datetime | None = None  # real time the plate was last read, when that differs (LEFT)
    last_read_snapshot: str | None = None  # the frame of that last read
    started: float = 0.0  # time.perf_counter() when handling began, to time the light

    @property
    def departed_at(self) -> datetime:
        """When the car was last seen: the time a stay ends and its bill runs to."""
        return self.last_read_at or self.at

    @property
    def attachment(self) -> str:
        """The frame attached to alerts: the last sighting for LEFT, else the event's frame."""
        return self.last_read_snapshot or self.snapshot


def dollars(cents: int) -> str:
    return f"${cents / 100:.2f}"


def rate_text(cents_per_hour: int) -> str:
    return f"{dollars(cents_per_hour)}/hour"


def bill_cents(
    arrived_at: datetime, left_at: datetime, rate_cents_per_hour: int, min_charge_cents: int = 0
) -> tuple[int, int]:
    """(started minutes, amount in cents) for a stay.

    Every started minute counts, with a minimum of one, the amount is rounded
    up to the next whole cent, and it is never below min_charge_cents.
    """
    seconds = max(0.0, (left_at - arrived_at).total_seconds())
    minutes = max(1, math.ceil(seconds / 60 - 1e-9))
    amount = -(-minutes * rate_cents_per_hour // 60)  # integer ceiling division
    return minutes, max(amount, min_charge_cents)


def bill_for_session(
    db: Database, session: dict, min_charge_cents: int, overstay_fee_cents: int, words: WordBook | None = None
) -> dict:
    """Compute, store and describe the bill of a closed stay: the one place the arithmetic lives.

    Without an overstay fee the model is shown the total only; with one it gets
    the breakdown (parking, overstay_fee, total) so its message can give it.
    The time billed is given as words.
    """
    words = words or WordBook()
    booking = db.booking(session["booking_id"])
    arrived_at, left_at = parse(session["arrived_at"]), parse(session["left_at"])
    rate = booking["rate_cents_per_hour"]
    minutes, parking = bill_cents(arrived_at, left_at, rate, min_charge_cents)
    overstayed = bool(session["overstayed_at"])
    fee = overstay_fee_cents if overstayed else 0
    total = parking + fee
    db.set_amount(session["id"], total)
    result = {
        "session_id": session["id"],
        "time_billed": words.duration(minutes * 60),
        "rate": rate_text(rate),
        "minimum_charge_applied": bill_cents(arrived_at, left_at, rate)[1] < min_charge_cents,
        "overstayed": overstayed,
    }
    if overstayed:
        result["parking"] = dollars(parking)
        result["overstay_fee"] = dollars(fee)
    result["total"] = dollars(total)
    result["amount_cents"] = total
    return result


def public(booking: dict, words: WordBook, now: datetime) -> dict:
    """A booking as shown to the model: no plate (it knows it), times as words, the rate in dollars."""
    return {
        "id": booking["id"],
        "driver": booking["driver"],
        "starts": words.clock(parse(booking["starts_at"]), now),
        "ends": words.clock(parse(booking["ends_at"]), now),
        "rate": rate_text(booking["rate_cents_per_hour"]),
    }


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


PLATE_PARAM = {"plate": {"type": "string", "description": "The plate from the event."}}
TOOL_SCHEMAS = [
    _tool(
        "lookup_booking",
        "Look up the booking for the plate. Returns status active, overstaying (the stay is "
        "open but its booking has ended), outside_window or no_booking, the booking if there "
        "is one, and any open stay.",
        PLATE_PARAM,
        ["plate"],
    ),
    _tool("start_session", "Start the stay of a car whose booking is active.", PLATE_PARAM, ["plate"]),
    _tool("end_session", "End the open stay of the plate.", PLATE_PARAM, ["plate"]),
    _tool(
        "compute_bill",
        "Compute and record the bill for an ended stay. Returns the total in dollars, and "
        "the breakdown (parking, overstay fee) when an overstay fee applied.",
        {"session_id": {"type": "integer", "description": "The stay's session_id."}},
        ["session_id"],
    ),
    _tool(
        "set_light",
        "Set the driveway signal light.",
        {"color": {"type": "string", "enum": list(COLORS)}},
        ["color"],
    ),
    _tool(
        "send_alert",
        "Send a message to the owner or to the booked driver. The event's snapshot photo is attached by the code.",
        {
            "to": {"type": "string", "enum": ["owner", "driver"]},
            "message": {"type": "string", "description": "One or two sentences."},
        },
        ["to", "message"],
    ),
]
TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]


class Toolbox:
    """The six tools, bound to the event they are allowed to act on."""

    def __init__(
        self,
        db: Database,
        light: Light,
        alerts: Alerts,
        event: EventContext,
        min_charge_cents: int = DEFAULT_MIN_CHARGE_CENTS,
        overstay_fee_cents: int = DEFAULT_OVERSTAY_FEE_CENTS,
        words: WordBook | None = None,
    ) -> None:
        self.db = db
        self.light = light
        self.alerts = alerts
        self.event = event
        self.min_charge_cents = min_charge_cents
        self.overstay_fee_cents = overstay_fee_cents
        self.words = words or WordBook()

    def call(self, name: str, arguments: str) -> dict:
        """Run one tool call as the model made it. Always returns a dict."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        try:
            args = json.loads(arguments) if arguments else {}
        except ValueError:
            return {"error": "arguments are not valid JSON"}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        try:
            return getattr(self, name)(**args)
        except TypeError as err:
            return {"error": f"wrong arguments for {name}: {err}"}
        except Exception as err:  # a tool must never take the attendant down
            return {"error": f"{name} failed: {type(err).__name__}"}

    def _plate_error(self, plate) -> dict | None:
        if not isinstance(plate, str) or normalize(plate) != self.event.plate:
            return {"error": PLATE_RULE}
        return None

    def _open_stay(self) -> dict | None:
        stay = self.db.open_session(self.event.plate)
        if stay is None:
            return None
        return {
            "session_id": stay["id"],
            "arrived": self.words.clock(parse(stay["arrived_at"]), self.event.at),
            "ending_soon_warned": bool(stay["ending_soon_at"]),
            "overstayed": bool(stay["overstayed_at"]),
        }

    def lookup_booking(self, plate: str) -> dict:
        if error := self._plate_error(plate):
            return error
        open_stay = self._open_stay()
        active = self.db.booking_at(self.event.plate, self.event.at)
        if active:
            return {"status": "active", "booking": public(active, self.words, self.event.at), "open_session": open_stay}
        stay = self.db.open_session(self.event.plate)
        if stay:
            # A booked car still here after its booking ended is overstaying, not unbooked.
            return {
                "status": "overstaying",
                "booking": public(self.db.booking(stay["booking_id"]), self.words, self.event.at),
                "now": self.words.clock(self.event.at, self.event.at),
                "open_session": open_stay,
            }
        latest = self.db.latest_booking(self.event.plate)
        if latest:
            return {
                "status": "outside_window",
                "booking": public(latest, self.words, self.event.at),
                "now": self.words.clock(self.event.at, self.event.at),
                "open_session": None,
            }
        return {"status": "no_booking", "booking": None, "open_session": None}

    def start_session(self, plate: str) -> dict:
        if error := self._plate_error(plate):
            return error
        if stay := self.db.open_session(self.event.plate):
            return {"error": f"a stay is already open for this plate (session_id {stay['id']})"}
        booking = self.db.booking_at(self.event.plate, self.event.at)
        if booking is None:
            return {"error": "no booking covers this plate at this time, so a stay cannot start"}
        session_id = self.db.start_session(self.event.plate, booking["id"], self.event.at)
        return {
            "session_id": session_id,
            "booking_id": booking["id"],
            "driver": booking["driver"],
            "arrived": self.words.clock(self.event.at, self.event.at),
            "status": "open",
        }

    def end_session(self, plate: str) -> dict:
        if error := self._plate_error(plate):
            return error
        stay = self.db.open_session(self.event.plate)
        if stay is None:
            return {"error": "there is no open stay for this plate"}
        left_at = self.event.departed_at
        self.db.close_session(stay["id"], left_at, self.event.attachment)
        arrived_at = parse(stay["arrived_at"])
        return {
            "session_id": stay["id"],
            "arrived": self.words.clock(arrived_at, self.event.at),
            "left": self.words.clock(left_at, self.event.at),
            "stayed": self.words.duration((left_at - arrived_at).total_seconds()),
            "status": "closed",
        }

    def compute_bill(self, session_id) -> dict:
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            return {"error": "session_id must be an integer"}
        session = self.db.session(session_id)
        if session is None:
            return {"error": f"there is no stay with session_id {session_id}"}
        if session["plate"] != self.event.plate:
            return {"error": PLATE_RULE}
        if session["status"] != "closed":
            return {"error": "the stay is still open; end it before billing"}
        return bill_for_session(self.db, session, self.min_charge_cents, self.overstay_fee_cents, self.words)

    def set_light(self, color: str) -> dict:
        color = str(color).lower()
        if color not in COLORS:
            return {"error": f"color must be one of {', '.join(COLORS)}"}
        try:
            self.light.set(color)
        except Exception as err:  # the bulb did not respond; the finish check will try again
            return {"error": f"the light did not respond: {err}"}
        return {"light": color, "seconds_after_event": round(time.perf_counter() - self.event.started, 1)}

    def send_alert(self, to: str, message: str) -> dict:
        to = str(to).lower()
        if to not in ("owner", "driver"):
            return {"error": "to must be owner or driver"}
        if not isinstance(message, str) or not message.strip():
            return {"error": "message must not be empty"}
        name = None
        if to == "driver":
            booking = self.db.latest_booking(self.event.plate)
            if booking is None:
                return {"error": "no booking for this plate, so there is no driver to alert"}
            name = booking["driver"]
        error = self.alerts.send(Alert(to, name, message.strip(), self.event.attachment))
        return {
            "sent_to": to,
            "name": name,
            "snapshot": self.event.attachment,
            "delivered": error is None,
            "delivery_error": error,
        }
