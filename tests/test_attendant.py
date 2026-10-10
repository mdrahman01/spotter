"""Tests of the attendant's finish check with a scripted stand-in for the model: no network."""

import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from spotter.alerts import Alert, ConsoleAlerts
from spotter.attendant import AFTER_REMINDER, BY_CODE, CLEAN, REMINDER_ROUNDS, Attendant
from spotter.db import Database
from spotter.events import Event, EventFrames
from spotter.light import ConsoleLight, LightError
from spotter.privacy import Masker

PLATE = "TEST123"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
SNAP = "data/snapshots/a.jpg"
LOOKUP = ("lookup_booking", {"plate": PLATE})
START = ("start_session", {"plate": PLATE})
GREEN = ("set_light", {"color": "green"})
WELCOME = ("send_alert", {"to": "driver", "message": "Welcome, your stay has started."})


def quiet(line: str) -> None:
    pass


class Down(Exception):
    pass


class FakeModel:
    """Scripted replies: a list of (tool, args) is a round of tool calls, a str a closing sentence,
    an exception a failed call. When the script runs out it keeps saying "Done."."""

    api_key = "fake-key"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self.create)))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.script.pop(0) if self.script else "Done."
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            message = {"role": "assistant", "content": reply}
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(reply)
                ],
            }
        body = {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return SimpleNamespace(text=json.dumps(body))

    def user_messages(self) -> list[str]:
        return [m["content"] for m in self.requests[-1]["messages"] if m["role"] == "user"]


class FailingAlerts:
    """Every send fails, as Telegram would without a network; remembers the attempts."""

    def __init__(self):
        self.attempts: list[Alert] = []

    def send(self, alert: Alert) -> str | None:
        self.attempts.append(alert)
        return "URLError: simulated outage"


class DeadLight:
    """A bulb that never answers."""

    color = None

    def __init__(self):
        self.attempts = 0

    def set(self, color):
        self.attempts += 1
        raise LightError("OSError: no route to host")


class Scene:
    def __init__(self, script, booked=True, alerts=None, light=None):
        self.db = Database(":memory:")
        if booked:
            self.booking = self.db.add_booking(PLATE, "Dana Driver", T0 - timedelta(hours=1), T0 + timedelta(hours=11), 500)
        self.light = light if light is not None else ConsoleLight(say=quiet)
        self.alerts = alerts if alerts is not None else ConsoleAlerts(say=quiet)
        self.model = FakeModel(script)
        masker = Masker(None)
        masker.print = quiet
        self.attendant = Attendant(self.model, self.db, self.light, self.alerts, masker, min_charge_cents=100, overstay_fee_cents=500)

    def arrive(self):
        self.attendant.handle(Event("ARRIVED", PLATE, 6.6, 5.8, 6.6, 5), EventFrames(T0, SNAP, T0, SNAP))

    def leave(self, seconds_after=26.2, last_read_after=22.2):
        at, last = T0 + timedelta(seconds=seconds_after), T0 + timedelta(seconds=last_read_after)
        self.attendant.handle(Event("LEFT", PLATE, seconds_after, 5.8, last_read_after, 81), EventFrames(at, SNAP, last, "data/snapshots/last.jpg"))

    def rows(self, tool=None):
        return [a for a in self.db.actions() if tool is None or a["tool"] == tool]

    def by_code(self):
        return [a["tool"] for a in self.db.actions() if json.loads(a["result"]).get("by") == "code"]

    def finish(self):
        return json.loads(self.rows("finish")[-1]["result"])


class FinishCheckTest(unittest.TestCase):
    def test_a_clean_event_gets_no_reminder(self):
        scene = Scene([[LOOKUP], [START], [GREEN], [WELCOME], "Started the stay, lit green, alerted the driver."])
        scene.arrive()
        self.assertEqual(scene.rows("reminder"), [])
        self.assertEqual(scene.by_code(), [])
        self.assertEqual(scene.attendant.stats.finished[CLEAN], 1)
        self.assertEqual(scene.light.color, "green")
        self.assertEqual(json.loads(scene.rows("finish")[0]["arguments"])["how"], CLEAN)

    def test_the_light_skipped_then_set_after_the_reminder(self):
        scene = Scene([[LOOKUP], [START], [WELCOME], "Done.", [GREEN], "Lit green."])
        scene.arrive()
        reminders = scene.rows("reminder")
        self.assertEqual(len(reminders), 1)
        told = json.loads(reminders[0]["result"])["told"]
        self.assertIn("set the light to green", told)
        self.assertNotIn("alert", told)  # only what was missing
        self.assertEqual(json.loads(reminders[0]["result"])["rounds"], REMINDER_ROUNDS)
        self.assertEqual(scene.light.color, "green")
        self.assertEqual(scene.by_code(), [])
        self.assertEqual(scene.attendant.stats.finished[AFTER_REMINDER], 1)
        self.assertEqual(len(scene.model.requests), 6)  # 4 rounds, the closing sentence, 2 more after the reminder
        self.assertEqual(scene.model.user_messages()[-1][:13], "Finish check:")

    def test_the_reminder_ignored_then_code_sets_the_light(self):
        scene = Scene([[LOOKUP], [START], [WELCOME], "Done.", "Nothing more to do."])
        scene.arrive()
        self.assertEqual(len(scene.rows("reminder")), 1)
        self.assertEqual(scene.by_code(), ["set_light"])
        self.assertEqual(scene.light.color, "green")
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 1)
        self.assertEqual(scene.attendant.stats.code_items, 1)
        self.assertEqual(scene.finish()["completed_by_code"], ["set the light to green"])

    def test_an_undeliverable_alert_is_recorded_once_and_does_not_loop(self):
        failing = FailingAlerts()
        scene = Scene([[LOOKUP], [START], [GREEN], [WELCOME], "Done.", [WELCOME], "Sent again."], alerts=failing)
        scene.arrive()
        self.assertEqual(len(failing.attempts), 3)  # twice by the model, once by code
        self.assertEqual(len(scene.model.requests), 7)
        self.assertEqual(scene.by_code(), ["send_alert"])
        code_alert = json.loads(scene.rows("send_alert")[-1]["result"])
        self.assertEqual((code_alert["delivered"], code_alert["by"]), (False, "code"))
        self.assertIn("simulated outage", code_alert["delivery_error"])
        self.assertEqual(scene.attendant.stats.undeliverable, 1)
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 1)

    def test_an_offline_model_is_completed_by_code_without_a_reminder(self):
        scene = Scene([Down("api down"), Down("still down")])
        scene.arrive()
        self.assertEqual((scene.attendant.stats.model_failures, scene.attendant.stats.offline_events), (2, 1))
        self.assertEqual(scene.rows("reminder"), [])
        self.assertEqual(scene.by_code(), ["start_session", "set_light", "send_alert"])
        self.assertIsNotNone(scene.db.open_session(PLATE))
        self.assertEqual(scene.light.color, "green")
        self.assertEqual([a.to for a in scene.alerts.sent], ["driver"])
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 1)

    def test_running_out_of_rounds_gets_a_reminder_then_code(self):
        scene = Scene([[LOOKUP]] * 10 + ["Done."])
        scene.arrive()
        self.assertEqual(scene.attendant.stats.rounds_exhausted, 1)
        self.assertEqual(len(scene.rows("reminder")), 1)
        self.assertEqual(scene.by_code(), ["start_session", "set_light", "send_alert"])
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 1)

    def test_left_after_an_overstay_needs_the_owner_told_what_was_billed(self):
        scene = Scene([])
        scene.db.start_session(PLATE, scene.booking, T0)
        scene.db.mark_overstayed(1, T0 + timedelta(seconds=15))
        bill_alert = ("send_alert", {"to": "driver", "message": "Your stay has ended. The bill is $6.00: parking $1.00, overstay fee $5.00."})
        scene.model.script = [[LOOKUP], [("end_session", {"plate": PLATE})], [("compute_bill", {"session_id": 1})], [bill_alert], "Billed the driver.", "Nothing more."]
        scene.leave()
        told = json.loads(scene.rows("reminder")[0]["result"])["told"]
        self.assertIn("owner", told)
        self.assertIn("set the light to off", told)
        self.assertNotIn("driver", told)  # the driver alert with the total was delivered
        self.assertEqual(scene.by_code(), ["send_alert", "set_light"])
        owner = [a for a in scene.alerts.sent if a.to == "owner"][-1]
        self.assertIn("$6.00", owner.message)
        self.assertEqual(owner.snapshot, "data/snapshots/last.jpg")
        self.assertEqual(scene.light.color, "off")
        session = scene.db.session(1)
        self.assertEqual((session["status"], session["left_at"], session["amount_cents"]), ("closed", "2026-01-01T12:00:22+00:00", 600))

    def test_left_with_nothing_done_is_closed_billed_and_told_by_code(self):
        scene = Scene([[LOOKUP], "Done.", "Still nothing."])
        scene.db.start_session(PLATE, scene.booking, T0)
        scene.leave()
        self.assertEqual(scene.by_code(), ["end_session", "compute_bill", "send_alert", "set_light"])
        session = scene.db.session(1)
        self.assertEqual((session["status"], session["amount_cents"], session["left_snapshot"]), ("closed", 100, "data/snapshots/last.jpg"))
        self.assertIn("$1.00", scene.alerts.sent[-1].message)
        self.assertEqual(scene.light.color, "off")

    def test_a_light_that_does_not_respond_is_reported_once_and_the_owner_told_once(self):
        dead = DeadLight()
        scene = Scene([[LOOKUP], [START], [GREEN], [WELCOME], "Done.", [GREEN], "Tried again."], light=dead)
        scene.arrive()
        tool_errors = [json.loads(a["result"]) for a in scene.rows("set_light")]
        self.assertTrue(all("did not respond" in r["error"] for r in tool_errors))
        self.assertEqual(len(tool_errors), 3)  # the model twice, code once
        self.assertEqual(scene.by_code(), ["set_light", "send_alert"])
        owner = [a for a in scene.alerts.sent if a.to == "owner"]
        self.assertEqual(len(owner), 1)
        self.assertIn("not responding", owner[0].message)
        self.assertEqual(dead.attempts, 3)
        # A second event with the same dead light: recorded again, but the owner is not told twice.
        scene.model.script = [[LOOKUP], [("end_session", {"plate": PLATE})], [("compute_bill", {"session_id": 1})],
                              [("send_alert", {"to": "driver", "message": "Your stay has ended. The bill is $1.00."})], "Done.", "Nothing more."]
        scene.leave()
        self.assertEqual(len([a for a in scene.alerts.sent if a.to == "owner"]), 1)
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 2)

    def test_the_model_is_given_times_as_words_and_they_are_recorded(self):
        scene = Scene([[LOOKUP], [START], [GREEN], [WELCOME], "Done."])
        scene.arrive()
        payload = json.loads(scene.model.requests[0]["messages"][1]["content"])
        self.assertRegex(payload["time"], r"\d{1,2}:\d{2} (am|pm)")
        self.assertNotIn("T", payload["time"])
        lookup = json.loads(next(m for m in scene.model.requests[1]["messages"] if m["role"] == "tool")["content"])
        self.assertRegex(lookup["booking"]["starts"], r"\d{1,2}:\d{2} (am|pm)")
        self.assertNotIn("starts_at", lookup["booking"])
        words = json.loads(scene.rows("words")[0]["arguments"])["supplied"]
        self.assertIn(payload["time"], words)
        self.assertIn(lookup["booking"]["ends"], words)

    def test_an_unbooked_arrival_and_departure(self):
        scene = Scene([[LOOKUP], [("set_light", {"color": "red"})], "Done.", "Nothing more."], booked=False)
        scene.arrive()
        self.assertEqual(scene.by_code(), ["send_alert"])
        self.assertEqual(scene.alerts.sent[-1].to, "owner")
        self.assertIsNone(scene.db.open_session(PLATE))
        scene.model.script = [[LOOKUP], [("send_alert", {"to": "owner", "message": "The car has gone after 16 seconds."})], "Told the owner."]
        scene.leave()
        self.assertEqual(scene.by_code(), ["send_alert", "set_light"])  # the light-off of LEFT was done by code
        self.assertEqual(scene.light.color, "off")
        self.assertEqual(scene.attendant.stats.finished[BY_CODE], 2)


if __name__ == "__main__":
    unittest.main()
