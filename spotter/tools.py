"""The only things the model can do: six plain functions, each with an OpenAI tool schema.

The code, not the model, enforces the rules and does the arithmetic:
- a tool acts only on the plate of the current event;
- a plate has one open stay at a time, and a stay needs a booking covering its start;
- the bill is per started minute at the booking's rate, with a one-minute
  minimum, rounded up to whole cents.
A broken rule comes back to the model as {"error": ...}; nothing raises. Error
messages never contain plate text, so they can be printed as they are.
"""

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime

from spotter.alerts import Alert, Alerts
from spotter.db import Database, iso, parse
from spotter.light import COLORS, Light
from spotter.reader import normalize

PLATE_RULE = "this tool may only act on the plate in the current event"


@dataclass(frozen=True)
class EventContext:
    """The event the tools act on."""

    type: str  # ARRIVED or LEFT
    plate: str
    at: datetime  # the event's real time
    snapshot: str  # repo-relative path of the event's frame
    started: float = 0.0  # time.perf_counter() when handling began, to time the light


def bill_cents(arrived_at: datetime, left_at: datetime, rate_cents_per_hour: int) -> tuple[int, int]:
    """(started minutes, amount in cents) for a stay.

    Every started minute counts, with a minimum of one, and the amount is
    rounded up to the next whole cent.
    """
    seconds = max(0.0, (left_at - arrived_at).total_seconds())
    minutes = max(1, math.ceil(seconds / 60 - 1e-9))
    amount = -(-minutes * rate_cents_per_hour // 60)  # integer ceiling division
    return minutes, amount


def public(booking: dict) -> dict:
    """A booking as shown to the model: everything but the plate, which it already knows."""
    keys = ("id", "driver", "starts_at", "ends_at", "rate_cents_per_hour")
    return {key: booking[key] for key in keys}


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
        "Look up the booking for the plate. Returns status active, outside_window or "
        "no_booking, the booking if there is one, and any open stay.",
        PLATE_PARAM,
        ["plate"],
    ),
    _tool("start_session", "Start the stay of a car whose booking is active.", PLATE_PARAM, ["plate"]),
    _tool("end_session", "End the open stay of the plate.", PLATE_PARAM, ["plate"]),
    _tool(
        "compute_bill",
        "Compute and record the bill for an ended stay.",
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
        "Send a message to the owner or to the booked driver. The event's snapshot is attached by the code.",
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

    def __init__(self, db: Database, light: Light, alerts: Alerts, event: EventContext) -> None:
        self.db = db
        self.light = light
        self.alerts = alerts
        self.event = event

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
        return {"session_id": stay["id"], "arrived_at": stay["arrived_at"]} if stay else None

    def lookup_booking(self, plate: str) -> dict:
        if error := self._plate_error(plate):
            return error
        active = self.db.booking_at(self.event.plate, self.event.at)
        if active:
            return {"status": "active", "booking": public(active), "open_session": self._open_stay()}
        latest = self.db.latest_booking(self.event.plate)
        if latest:
            return {
                "status": "outside_window",
                "booking": public(latest),
                "now": iso(self.event.at),
                "open_session": self._open_stay(),
            }
        return {"status": "no_booking", "booking": None, "open_session": self._open_stay()}

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
            "arrived_at": iso(self.event.at),
            "status": "open",
        }

    def end_session(self, plate: str) -> dict:
        if error := self._plate_error(plate):
            return error
        stay = self.db.open_session(self.event.plate)
        if stay is None:
            return {"error": "there is no open stay for this plate"}
        self.db.close_session(stay["id"], self.event.at)
        return {
            "session_id": stay["id"],
            "arrived_at": stay["arrived_at"],
            "left_at": iso(self.event.at),
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
        booking = self.db.booking(session["booking_id"])
        minutes, amount = bill_cents(
            parse(session["arrived_at"]), parse(session["left_at"]), booking["rate_cents_per_hour"]
        )
        self.db.set_amount(session_id, amount)
        return {
            "session_id": session_id,
            "minutes": minutes,
            "rate_cents_per_hour": booking["rate_cents_per_hour"],
            "amount_cents": amount,
            "amount": f"${amount / 100:.2f}",
        }

    def set_light(self, color: str) -> dict:
        color = str(color).lower()
        if color not in COLORS:
            return {"error": f"color must be one of {', '.join(COLORS)}"}
        self.light.set(color)
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
        self.alerts.send(Alert(to, name, message.strip(), self.event.snapshot))
        return {"sent_to": to, "name": name, "snapshot": self.event.snapshot}
