"""Check what the attendant did in one scenario and print PASS or FAIL.

    .venv/bin/python scripts/check_attendant.py booked|unknown|overstay|unreadable [--since ISO_TIME] [--db PATH]

With --since only the events and stays from that time on are checked (a time
without a zone is UTC), so one scene of a long live run can be checked on its
own; --db checks a copy. The run's own rows (RUN_STARTED, RUN_STOPPED, camera
outages) are noted and skipped.

Reads data/spotter.db. For every event it works out the end state the rules
require, the same facts the attendant's finish check uses, from the sessions
table and the set_light and send_alert calls in action_log, whether the model
made them or code completed them, and prints how the event finished: by the
attendant, after a reminder, or completed by code. Only delivered alerts count.
Alerts beyond the required ones are listed as notes, not failures. Prints no
plate text and no alert messages.

Required end state:
  ARRIVED, active booking: a stay opened at the event time, light green, a driver alert.
  ARRIVED, no booking:     no stay, light red, an owner alert.
  ENDING_SOON:             a driver alert containing the time-left words code supplied.
  OVERSTAY:                light amber, a driver alert containing the fee amount, an owner alert.
  LEFT, open stay:         the stay closed at the last read with its departure snapshot, the
                           bill stored, a driver alert containing the total, an owner alert
                           containing how long the car stayed and the total, light off.
  LEFT, no stay:           an owner alert, light off.
  UNREADABLE_VEHICLE:      light red, an owner alert.
  SPOT_CLEAR:              light off, an owner alert containing how long the vehicle stayed.
For every LEFT the seconds from the last read to the light going off are
reported, and a note is made when a message went out before the light changed
(the rule is light first). Photos follow the messages from a background
thread; one that was not delivered is noted.
Wording: in booked and overstay no closing sentence or alert may call the car
unbooked; in booked no alert may mention an overstay; and no alert may state a
clock time or a duration that code did not supply for that event (the "words"
row of the event lists what code supplied). Shape: booked has one
closed session billed at least the minimum charge; unknown has no session;
overstay has ARRIVED, ENDING_SOON, OVERSTAY, LEFT in that order and a 600 cent
bill with a 500 cent overstay fee; unreadable has UNREADABLE_VEHICLE then
SPOT_CLEAR, no ARRIVED or LEFT, and no session.
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.db import Database, iso, parse  # noqa: E402
from spotter.timers import time_left_words  # noqa: E402
from spotter.tools import dollars  # noqa: E402
from spotter.words import duration_words  # noqa: E402

SCENARIOS = ("booked", "unknown", "overstay", "unreadable")
OVERSTAY_EVENTS = ["ARRIVED", "ENDING_SOON", "OVERSTAY", "LEFT"]
CODE_EVENTS = ("CAMERA_OFFLINE", "CAMERA_BACK", "RUN_STARTED", "RUN_STOPPED")  # code's own rows, noted and skipped

CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\b", re.IGNORECASE)
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
TIME_LEFT = re.compile(r"less than a minute|about \d+ minutes?")  # the words time_left_words makes
DURATION = re.compile(r"\b(\d+(?:\.\d+)?)[\s-]*(second|sec|minute|min|hour|hr|day)s?\b", re.IGNORECASE)
UNITS = {"sec": "second", "min": "minute", "hr": "hour"}


def time_tokens(text: str) -> set[str]:
    """Every clock time or duration stated in `text`, one canonical token each."""
    tokens = set()
    for hours, minutes, meridiem in CLOCK.findall(text):
        tokens.add(f"clock {int(hours)}:{minutes} {meridiem.lower()}".rstrip())
    for match in ISO.findall(text):
        tokens.add(f"iso {match}")
    for number, unit in DURATION.findall(text):
        unit = unit.lower()
        tokens.add(f"duration {float(number):g} {UNITS.get(unit, unit)}")
    return tokens


def load_json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {}


def option(args: list[str], flag: str) -> str | None:
    """Take `flag VALUE` out of args and return VALUE, or None when the flag is absent.
    Raises ValueError when the flag is the last argument, so the caller prints the usage line."""
    if flag not in args:
        return None
    i = args.index(flag)
    if i + 1 >= len(args):
        raise ValueError(f"{flag} needs a value")
    value = args[i + 1]
    del args[i:i + 2]
    return value


def main() -> int:
    args = [a for a in sys.argv[1:]]
    since = db_path = None
    try:
        since, db_path = option(args, "--since"), option(args, "--db")
    except ValueError:
        args = []  # the usage line follows
    if len(args) != 1 or args[0] not in SCENARIOS:
        print("usage: check_attendant.py booked|unknown|overstay|unreadable [--since ISO_TIME] [--db PATH]", file=sys.stderr)
        return 2
    scenario = args[0]
    settings = config.load({})
    fee_text = dollars(settings.overstay_fee_cents)
    db = Database(db_path) if db_path else Database()
    sessions = db.sessions()
    if since:
        moment = parse(since)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        since_iso = iso(moment)  # the stored form, whatever zone was typed, so the strings compare in time order
        sessions = [s for s in sessions if s["arrived_at"] >= since_iso]

    # Group the action log into events, in order.
    groups: list[dict] = []
    for action in db.actions():
        event = json.loads(action["event"])
        if since and (event.get("time") or "") < since_iso:
            continue
        if not groups or groups[-1]["event"] != event:
            groups.append({"event": event, "rows": []})
        groups[-1]["rows"].append(action)

    failures, notes, texts, alert_texts = [], [], [], []
    unsupplied: list[str] = []  # events whose alerts state a time or duration code did not supply
    finishes: Counter = Counter()
    light = None  # carried from event to event, as the real light is
    light_sequence = []
    bills = []
    model_problems = []
    for group in groups:
        event = group["event"]
        kind, at, plate = event["type"], parse(event["time"]), event.get("plate")
        if kind in CODE_EVENTS:
            notes.append(f"{kind} at {event['time']}")
            continue
        alerts, reminders, code_items, errors = [], 0, 0, []
        supplied_words: list[str] = []
        stayed_words: str | None = None  # from end_session: how long the car stayed, as code put it
        light_off_at: datetime | None = None  # LEFT: when the light went off
        message_before_light = False
        for row in group["rows"]:
            result = load_json(row["result"])
            arguments = load_json(row["arguments"])
            by_code = result.get("by") == "code" or arguments.get("by") == "code"
            tool = row["tool"]
            if tool == "reminder":
                reminders += 1
                continue
            if tool == "finish":
                continue
            if tool == "words":
                supplied_words = arguments.get("supplied") or []
                continue
            if tool == "final":
                texts.append(result.get("text") or "")
                continue
            if tool == "model":
                model_problems.append((kind, result.get("error")))
                continue
            if tool == "venue_search":  # the price check's own searches, run by code before the judge
                continue
            if tool == "photo":
                if not result.get("delivered"):
                    notes.append(f"{kind}: photos for the {arguments.get('to')} not delivered: {result.get('error')}")
                continue
            if "error" in result:
                errors.append((tool, result["error"]))
                continue
            code_items += by_code
            if tool == "set_light":
                light = arguments.get("color") or result.get("light")
                light_sequence.append(light)
                if kind == "LEFT" and light == "off" and light_off_at is None:
                    light_off_at = parse(row["time"])
            elif tool == "send_alert":
                message = arguments.get("message") or ""
                alerts.append((arguments.get("to"), message, bool(result.get("delivered"))))
                texts.append(message)
                alert_texts.append(message)
                if kind == "LEFT" and light_off_at is None:
                    message_before_light = True
            elif tool == "compute_bill":
                bills.append(result)
            elif tool == "end_session":
                stayed_words = result.get("stayed")

        delivered = [(to, message) for to, message, ok in alerts if ok]

        def has(to: str, needle: str | None = None) -> bool:
            return any(t == to and (needle is None or needle in m) for t, m in delivered)

        # The stay this event is about: the latest one opened at or before it.
        stay = next(
            (s for s in reversed(sessions) if s["plate"] == plate and s["arrived_at"] <= event["time"]), None
        )
        required: list[tuple[str, bool]] = []
        expected_recipients: set[str] = set()
        if kind == "ARRIVED":
            booking = db.booking_at(plate, at)
            opened_here = stay is not None and stay["arrived_at"] == event["time"]
            if booking:
                required = [("stay opened", opened_here), ("light green", light == "green"), ("driver alert", has("driver"))]
                expected_recipients = {"driver"}
            else:
                required = [("no stay", not opened_here), ("light red", light == "red"), ("owner alert", has("owner"))]
                expected_recipients = {"owner"}
        elif kind == "ENDING_SOON":
            # The words code supplied for this event (the run measures the time left at the exact
            # sighting); only without a words row is it worked out from the event time, to the second.
            words = next((w for w in supplied_words if TIME_LEFT.fullmatch(w)), None)
            if words is None and stay:
                ends_at = parse(db.booking(stay["booking_id"])["ends_at"])
                words = time_left_words((ends_at - at).total_seconds())
            required = [(f"driver alert with the time left ({words})", words is not None and has("driver", words))]
            expected_recipients = {"driver"}
        elif kind == "OVERSTAY":
            required = [
                ("light amber", light == "amber"),
                (f"driver alert with the fee {fee_text}", has("driver", fee_text)),
                ("owner alert", has("owner")),
            ]
            expected_recipients = {"driver", "owner"}
        elif kind == "UNREADABLE_VEHICLE":
            required = [("light red", light == "red"), ("owner alert", has("owner"))]
            expected_recipients = {"owner"}
        elif kind == "SPOT_CLEAR":
            stayed = event.get("stayed_for")
            required = [
                ("light off", light == "off"),
                (f"owner alert with how long it stayed ({stayed})", stayed is not None and has("owner", stayed)),
            ]
            expected_recipients = {"owner"}
        elif kind == "LEFT":
            last_read = event.get("last_read_at")
            if stay and (stay["left_at"] is None or stay["left_at"] >= (last_read or event["time"])):
                total = dollars(stay["amount_cents"]) if stay["amount_cents"] is not None else None
                if stayed_words is None and stay["left_at"]:
                    stayed_words = duration_words((parse(stay["left_at"]) - parse(stay["arrived_at"])).total_seconds())
                required = [
                    ("stay closed", stay["status"] == "closed"),
                    ("left_at is the last read", stay["left_at"] == last_read),
                    ("departure snapshot stored", bool(stay["left_snapshot"])),
                    ("bill stored", total is not None),
                    (f"driver alert with the total {total}", total is not None and has("driver", total)),
                    (f"owner alert with the stay length ({stayed_words}) and the total {total}",
                     total is not None and stayed_words is not None and has("owner", total) and has("owner", stayed_words)),
                    ("light off", light == "off"),
                ]
                expected_recipients = {"driver", "owner"}
            else:
                required = [("owner alert", has("owner")), ("light off", light == "off")]
                expected_recipients = {"owner"}
            if light_off_at is not None and last_read:
                seconds = (light_off_at - parse(last_read)).total_seconds()
                if seconds >= 0:
                    notes.append(f"LEFT: last read to light off {seconds:.0f}s")
                else:
                    notes.append("LEFT: last read to light off not measured (a replay runs faster than real time)")
            if message_before_light:
                notes.append("LEFT: a message went out before the light changed (the rule is light first)")

        allowed = set().union(*(time_tokens(words) for words in supplied_words)) if supplied_words else set()
        stated = set().union(*(time_tokens(message) for _, message, _ in alerts)) if alerts else set()
        if stated - allowed:
            unsupplied.append(f"{kind}: {sorted(stated - allowed)} (code supplied {sorted(allowed) or 'nothing'})")

        how = "completed by code" if code_items else ("after a reminder" if reminders else "by the attendant")
        finishes[how] += 1
        state = ", ".join(f"{label} {'ok' if ok else 'MISSING'}" for label, ok in required)
        print(f"{kind}: finished {how} | {state}")
        failures += [f"{kind}: {label}" for label, ok in required if not ok]
        recipients = Counter(to for to, _ in delivered)
        extra = [f"{to} x{n}" for to, n in recipients.items() if to not in expected_recipients or n > 1]
        if extra:
            notes.append(f"{kind}: more alerts than required ({', '.join(extra)})")
        if errors:
            notes.append(f"{kind}: tool errors returned to the model: {errors}")

    print("sessions:")
    for s in sessions:
        print(
            f"  id {s['id']} | booking {s['booking_id']} | arrived {s['arrived_at']} | left {s['left_at']}"
            f" | {s['status']} | amount_cents {s['amount_cents']} | ending_soon_at {s['ending_soon_at']}"
            f" | overstayed_at {s['overstayed_at']} | left_snapshot {'yes' if s['left_snapshot'] else 'no'}"
        )
    if not sessions:
        print("  none")
    for bill in bills:
        print(
            f"bill: total {bill.get('total')} ({bill.get('amount_cents')} cents)"
            + (f" | parking {bill.get('parking')} | overstay fee {bill.get('overstay_fee')}" if bill.get("overstayed") else "")
        )
    print(f"light changes, in order: {light_sequence or 'none'}")
    print(f"finish: {finishes['by the attendant']} by the attendant, {finishes['after a reminder']} after a reminder, {finishes['completed by code']} completed by code")
    print(f"model problems: {model_problems or 'none'}")
    for note in notes:
        print(f"note: {note}")

    # Wording, checked on the text without printing it.
    if scenario in ("booked", "overstay") and any("unbooked" in text.lower() for text in texts):
        failures.append("a closing sentence or alert calls the car unbooked")
    if scenario == "booked" and any("overstay" in text.lower() for text in alert_texts):
        failures.append("an alert mentions an overstay")
    for entry in unsupplied:
        failures.append(f"an alert states a time or duration code did not supply: {entry}")

    # Scenario shape.
    order = [group["event"]["type"] for group in groups if group["event"]["type"] not in CODE_EVENTS]
    if scenario == "overstay":
        if order != OVERSTAY_EVENTS:
            failures.append(f"events {order} are not {OVERSTAY_EVENTS}")
        if len(sessions) != 1:
            failures.append(f"{len(sessions)} sessions instead of 1")
        elif sessions[0]["amount_cents"] != 600:
            failures.append(f"bill is {sessions[0]['amount_cents']} cents instead of 600")
        if not any(b.get("overstay_fee") == "$5.00" and b.get("total") == "$6.00" for b in bills):
            failures.append("no bill shows a $5.00 overstay fee and a $6.00 total")
    elif scenario == "booked":
        if len(sessions) != 1:
            failures.append(f"{len(sessions)} sessions instead of 1")
        elif sessions[0]["amount_cents"] is None or sessions[0]["amount_cents"] < settings.min_charge_cents:
            failures.append(f"bill {sessions[0]['amount_cents']} is below the minimum charge {settings.min_charge_cents}")
    elif scenario == "unknown" and sessions:
        failures.append(f"{len(sessions)} sessions instead of none")
    elif scenario == "unreadable":
        if order != ["UNREADABLE_VEHICLE", "SPOT_CLEAR"]:
            failures.append(f"events {order} are not ['UNREADABLE_VEHICLE', 'SPOT_CLEAR']")
        if sessions:
            failures.append(f"{len(sessions)} sessions instead of none")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
