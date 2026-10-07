"""Check what the attendant did in one scenario and print PASS or FAIL.

    .venv/bin/python scripts/check_attendant.py booked
    .venv/bin/python scripts/check_attendant.py unknown

Reads data/spotter.db: the sessions table, and the set_light and send_alert
calls recorded in action_log. Only delivered alerts count. Prints no plate text
and no alert messages.

booked passes if there is exactly one session, closed, billed at least the
minimum charge; the light went green then off; and the driver got two alerts.
unknown passes if there are no sessions and no bill; the light went red then
off; and the owner got two alerts, on arrival and then on leaving.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.db import Database  # noqa: E402

EXPECTED_LIGHTS = {"booked": ["green", "off"], "unknown": ["red", "off"]}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in EXPECTED_LIGHTS:
        print("usage: check_attendant.py booked|unknown", file=sys.stderr)
        return 2
    scenario = sys.argv[1]
    min_charge = config.load({}).min_charge_cents
    db = Database()
    sessions = db.sessions()
    actions = db.actions()

    lights, alerts, undelivered, errors, model_problems = [], [], [], [], []
    for action in actions:
        event_type = json.loads(action["event"])["type"]
        result = json.loads(action["result"])
        try:
            arguments = json.loads(action["arguments"])
        except ValueError:
            arguments = {}
        if action["tool"] == "model":
            model_problems.append(result.get("error"))
        if "error" in result:
            errors.append((event_type, action["tool"], result["error"]))
            continue
        if action["tool"] == "set_light":
            lights.append(arguments.get("color"))
        elif action["tool"] == "send_alert":
            if result.get("delivery_error"):
                undelivered.append((arguments.get("to"), event_type, result["delivery_error"]))
            else:
                alerts.append((arguments.get("to"), event_type))

    print("sessions:")
    for s in sessions:
        print(
            f"  id {s['id']} | booking {s['booking_id']} | arrived {s['arrived_at']}"
            f" | left {s['left_at']} | {s['status']} | amount_cents {s['amount_cents']}"
        )
    if not sessions:
        print("  none")
    print(f"light changes, in order: {lights or 'none'}")
    print(f"alerts delivered (to, during event): {alerts or 'none'}")
    print(f"alerts not delivered: {undelivered or 'none'}")
    print(f"tool errors returned to the model: {errors or 'none'}")
    print(f"model problems: {model_problems or 'none'}")

    failures = []
    if lights != EXPECTED_LIGHTS[scenario]:
        failures.append(f"light sequence {lights} is not {EXPECTED_LIGHTS[scenario]}")
    if scenario == "booked":
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
