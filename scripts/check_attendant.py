"""Check what the attendant did in one scenario and print PASS or FAIL.

    .venv/bin/python scripts/check_attendant.py booked
    .venv/bin/python scripts/check_attendant.py unknown

Reads data/spotter.db: the sessions table, and the set_light and send_alert
calls recorded in action_log. Only delivered alerts count. Prints no plate text
and no alert messages.

Wording checks: in booked and overstay no closing sentence or alert may call the
car unbooked; in booked no alert may mention an overstay; in overstay the
ENDING_SOON alert must contain the code's time-left words and the OVERSTAY
driver alert the fee amount. In every scenario with a stay, left_at must equal
the time the plate was last read.

booked passes if there is exactly one session, closed, billed at least the
minimum charge; the light went green then off; and the driver got two alerts.
unknown passes if there are no sessions and no bill; the light went red then
off; and the owner got two alerts, on arrival and then on leaving.
overstay passes if the events came as ARRIVED, ENDING_SOON, OVERSTAY, LEFT;
the light went green, amber, off; the driver got four alerts (welcome, ending
soon, overstay, bill) and the owner one; and the bill is 600 cents with a
500 cent overstay fee in the breakdown.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.db import Database, parse  # noqa: E402
from spotter.timers import time_left_words  # noqa: E402
from spotter.tools import dollars  # noqa: E402

EXPECTED_LIGHTS = {
    "booked": ["green", "off"],
    "unknown": ["red", "off"],
    "overstay": ["green", "amber", "off"],
}
OVERSTAY_EVENTS = ["ARRIVED", "ENDING_SOON", "OVERSTAY", "LEFT"]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in EXPECTED_LIGHTS:
        print("usage: check_attendant.py booked|unknown|overstay", file=sys.stderr)
        return 2
    scenario = sys.argv[1]
    min_charge = config.load({}).min_charge_cents
    db = Database()
    sessions = db.sessions()
    actions = db.actions()

    lights, alerts, undelivered, errors, model_problems, bills, order = [], [], [], [], [], [], []
    finals, messages, left_events = [], [], []  # messages: (to, event type, event time, text)
    for action in actions:
        event = json.loads(action["event"])
        event_type = event["type"]
        if event_type not in order:
            order.append(event_type)
        if event_type == "LEFT" and event not in left_events:
            left_events.append(event)
        result = json.loads(action["result"])
        try:
            arguments = json.loads(action["arguments"])
        except ValueError:
            arguments = {}
        if action["tool"] == "final":
            finals.append(result.get("text") or "")
        if action["tool"] == "send_alert":
            messages.append((arguments.get("to"), event_type, event["time"], arguments.get("message") or ""))
        if action["tool"] == "model":
            model_problems.append(result.get("error"))
        if "error" in result:
            errors.append((event_type, action["tool"], result["error"]))
            continue
        if action["tool"] == "compute_bill":
            bills.append(result)
        if action["tool"] == "set_light":
            lights.append(arguments.get("color"))
        elif action["tool"] == "send_alert":
            if result.get("delivery_error"):
                undelivered.append((arguments.get("to"), event_type, result["delivery_error"]))
            else:
                alerts.append((arguments.get("to"), event_type))

    print(f"events handled, in order: {order or 'none'}")
    print("sessions:")
    for s in sessions:
        print(
            f"  id {s['id']} | booking {s['booking_id']} | arrived {s['arrived_at']}"
            f" | left {s['left_at']} | {s['status']} | amount_cents {s['amount_cents']}"
            f" | ending_soon_at {s['ending_soon_at']} | overstayed_at {s['overstayed_at']}"
        )
    if not sessions:
        print("  none")
    for bill in bills:
        print(
            f"bill breakdown: parking {bill.get('parking')} | overstay fee {bill.get('overstay_fee')}"
            f" | total {bill.get('total')} ({bill.get('amount_cents')} cents)"
        )
    print(f"light changes, in order: {lights or 'none'}")
    print(f"alerts delivered (to, during event): {alerts or 'none'}")
    print(f"alerts not delivered: {undelivered or 'none'}")
    print(f"tool errors returned to the model: {errors or 'none'}")
    print(f"model problems: {model_problems or 'none'}")

    failures = []
    if lights != EXPECTED_LIGHTS[scenario]:
        failures.append(f"light sequence {lights} is not {EXPECTED_LIGHTS[scenario]}")

    # Wording, checked on the text without printing it.
    texts = finals + [text for _, _, _, text in messages]
    if scenario in ("booked", "overstay") and any("unbooked" in text.lower() for text in texts):
        failures.append("a closing sentence or alert calls the car unbooked")
    if scenario == "booked" and any("overstay" in text.lower() for _, _, _, text in messages):
        failures.append("an alert mentions an overstay")
    if scenario == "overstay" and sessions:
        booking = db.booking(sessions[0]["booking_id"])
        ends_at = parse(booking["ends_at"])
        warnings = [
            (text, time_left_words((ends_at - parse(when)).total_seconds()))
            for to, kind, when, text in messages
            if to == "driver" and kind == "ENDING_SOON"
        ]
        if not any(words in text for text, words in warnings):
            failures.append(
                "the ENDING_SOON alert does not contain the code's time-left words "
                f"({[words for _, words in warnings] or 'no such alert'})"
            )
        fee = dollars(config.load({}).overstay_fee_cents)
        if not any(fee in text for to, kind, _, text in messages if to == "driver" and kind == "OVERSTAY"):
            failures.append(f"the OVERSTAY driver alert does not state the fee {fee}")

    # A stay ends when the car was last seen, not when the rule noticed.
    for s in sessions:
        if s["status"] != "closed":
            continue
        last_read = next((event.get("last_read_at") for event in left_events), None)
        if s["left_at"] != last_read:
            failures.append(f"left_at {s['left_at']} is not the last-read time {last_read}")
        if not s["left_snapshot"]:
            failures.append("the session has no departure snapshot")
    if scenario == "overstay":
        if order != OVERSTAY_EVENTS:
            failures.append(f"events {order} are not {OVERSTAY_EVENTS}")
        driver_events = [event for to, event in alerts if to == "driver"]
        owner_events = [event for to, event in alerts if to == "owner"]
        if driver_events != OVERSTAY_EVENTS:
            failures.append(f"driver alerts during {driver_events} instead of one per event {OVERSTAY_EVENTS}")
        if owner_events != ["OVERSTAY"]:
            failures.append(f"owner alerts during {owner_events} instead of ['OVERSTAY']")
        if len(sessions) != 1:
            failures.append(f"{len(sessions)} sessions instead of 1")
        else:
            s = sessions[0]
            if s["status"] != "closed":
                failures.append("the session is not closed")
            if not s["overstayed_at"]:
                failures.append("the session is not marked overstayed")
            if s["amount_cents"] != 600:
                failures.append(f"bill is {s['amount_cents']} cents instead of 600")
        if len(bills) != 1 or bills[0].get("overstay_fee") != "$5.00" or bills[0].get("total") != "$6.00":
            failures.append(f"bill breakdown {bills} does not show a $5.00 overstay fee and a $6.00 total")
    elif scenario == "booked":
        if len(sessions) != 1:
            failures.append(f"{len(sessions)} sessions instead of 1")
        else:
            s = sessions[0]
            if s["status"] != "closed" or not s["left_at"]:
                failures.append("the session is not closed")
            if s["amount_cents"] is None:
                failures.append("no bill was computed")
            elif s["amount_cents"] < min_charge:
                failures.append(f"bill {s['amount_cents']} cents is below the minimum charge ({min_charge})")
        driver_alerts = sum(1 for to, _ in alerts if to == "driver")
        if driver_alerts != 2:
            failures.append(f"the driver got {driver_alerts} alerts instead of 2")
    else:
        if sessions:
            failures.append(f"{len(sessions)} sessions instead of none")
        if any(s["amount_cents"] is not None for s in sessions):
            failures.append("a bill was computed")
        owner_events = [event for to, event in alerts if to == "owner"]
        if owner_events != ["ARRIVED", "LEFT"]:
            failures.append(f"owner alerts during {owner_events} instead of ['ARRIVED', 'LEFT']")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
