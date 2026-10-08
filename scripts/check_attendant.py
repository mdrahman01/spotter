"""Check what the attendant did in one scenario and print PASS or FAIL.

    .venv/bin/python scripts/check_attendant.py booked|unknown|overstay

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
  ENDING_SOON:             a driver alert containing the time-left words.
  OVERSTAY:                light amber, a driver alert containing the fee amount, an owner alert.
  LEFT, open stay:         the stay closed at the last read with its departure snapshot, the
                           bill stored, a driver alert containing the total, light off; and if
                           the stay overstayed, an owner alert containing the total.
  LEFT, no stay:           an owner alert, light off.
Wording: in booked and overstay no closing sentence or alert may call the car
unbooked; in booked no alert may mention an overstay. Shape: booked has one
closed session billed at least the minimum charge; unknown has no session;
overstay has ARRIVED, ENDING_SOON, OVERSTAY, LEFT in that order and a 600 cent
bill with a 500 cent overstay fee.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.db import Database, parse  # noqa: E402
from spotter.timers import time_left_words  # noqa: E402
from spotter.tools import dollars  # noqa: E402

SCENARIOS = ("booked", "unknown", "overstay")
OVERSTAY_EVENTS = ["ARRIVED", "ENDING_SOON", "OVERSTAY", "LEFT"]


def load_json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print("usage: check_attendant.py booked|unknown|overstay", file=sys.stderr)
        return 2
    scenario = sys.argv[1]
    settings = config.load({})
    fee_text = dollars(settings.overstay_fee_cents)
    db = Database()
    sessions = db.sessions()

    # Group the action log into events, in order.
    groups: list[dict] = []
    for action in db.actions():
        event = json.loads(action["event"])
        if not groups or groups[-1]["event"] != event:
            groups.append({"event": event, "rows": []})
        groups[-1]["rows"].append(action)

    failures, notes, texts, alert_texts = [], [], [], []
    finishes: Counter = Counter()
    light = None  # carried from event to event, as the real light is
    light_sequence = []
    bills = []
    model_problems = []
    for group in groups:
        event = group["event"]
        kind, at, plate = event["type"], parse(event["time"]), event["plate"]
        alerts, reminders, code_items, errors = [], 0, 0, []
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
            if tool == "final":
                texts.append(result.get("text") or "")
                continue
            if tool == "model":
                model_problems.append((kind, result.get("error")))
                continue
            if "error" in result:
                errors.append((tool, result["error"]))
                continue
            code_items += by_code
            if tool == "set_light":
                light = arguments.get("color") or result.get("light")
                light_sequence.append(light)
            elif tool == "send_alert":
                message = arguments.get("message") or ""
                alerts.append((arguments.get("to"), message, bool(result.get("delivered"))))
                texts.append(message)
                alert_texts.append(message)
            elif tool == "compute_bill":
                bills.append(result)

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
            words = None
            if stay:
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
        elif kind == "LEFT":
            last_read = event.get("last_read_at")
            if stay and (stay["left_at"] is None or stay["left_at"] >= (last_read or event["time"])):
                total = dollars(stay["amount_cents"]) if stay["amount_cents"] is not None else None
                required = [
                    ("stay closed", stay["status"] == "closed"),
                    ("left_at is the last read", stay["left_at"] == last_read),
                    ("departure snapshot stored", bool(stay["left_snapshot"])),
                    ("bill stored", total is not None),
                    (f"driver alert with the total {total}", total is not None and has("driver", total)),
                    ("light off", light == "off"),
                ]
                expected_recipients = {"driver"}
                if stay["overstayed_at"]:
                    required.append((f"owner alert with the total {total}", total is not None and has("owner", total)))
                    expected_recipients.add("owner")
            else:
                required = [("owner alert", has("owner")), ("light off", light == "off")]
                expected_recipients = {"owner"}

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

    # Scenario shape.
    order = [group["event"]["type"] for group in groups]
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

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
