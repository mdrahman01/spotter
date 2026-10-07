"""Tests of the tool rules and the bill arithmetic: in-memory database, no network, no model."""

import unittest
from datetime import datetime, timedelta, timezone

from spotter.alerts import ConsoleAlerts
from spotter.db import Database
from spotter.light import ConsoleLight
from spotter.tools import EventContext, Toolbox, bill_cents

PLATE = "TEST123"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
SNAPSHOT = "data/snapshots/test.jpg"
RATE = 500  # cents per hour


def quiet(line: str) -> None:
    pass


class Scene:
    """A database, the stand-ins and a toolbox bound to one event."""

    def __init__(self, event_type="ARRIVED", plate=PLATE, at=T0, booked=True):
        self.db = Database(":memory:")
        if booked:
            self.booking_id = self.db.add_booking(
                PLATE, "Dana Driver", T0 - timedelta(hours=1), T0 + timedelta(hours=11), RATE
            )
        self.light = ConsoleLight(say=quiet)
        self.alerts = ConsoleAlerts(say=quiet)
        self.tools = self.rebind(event_type, plate, at)

    def rebind(self, event_type, plate=PLATE, at=T0) -> Toolbox:
        """A toolbox for a later event on the same database."""
        self.tools = Toolbox(self.db, self.light, self.alerts, EventContext(event_type, plate, at, SNAPSHOT))
        return self.tools


class BillArithmeticTest(unittest.TestCase):
    def bill(self, seconds, rate=RATE):
        return bill_cents(T0, T0 + timedelta(seconds=seconds), rate)

    def test_one_minute_minimum(self):
        self.assertEqual(self.bill(0), (1, 9))
        self.assertEqual(self.bill(10), (1, 9))
        self.assertEqual(self.bill(59.9), (1, 9))

    def test_every_started_minute_counts(self):
        self.assertEqual(self.bill(60), (1, 9))
        self.assertEqual(self.bill(61), (2, 17))
        self.assertEqual(self.bill(119), (2, 17))
        self.assertEqual(self.bill(7200), (120, 1000))

    def test_cents_round_up(self):
        self.assertEqual(self.bill(60, rate=1), (1, 1))  # 1/60 of a cent still costs a cent
        self.assertEqual(self.bill(3600, rate=1), (60, 1))
        self.assertEqual(self.bill(60, rate=0), (1, 0))

    def test_clock_going_backwards_is_billed_as_one_minute(self):
        self.assertEqual(bill_cents(T0, T0 - timedelta(minutes=5), RATE), (1, 9))


class ToolRulesTest(unittest.TestCase):
    def test_lookup_reports_active_outside_window_and_no_booking(self):
        self.assertEqual(Scene().tools.lookup_booking(PLATE)["status"], "active")
        late = Scene(at=T0 + timedelta(hours=20)).tools.lookup_booking(PLATE)
        self.assertEqual(late["status"], "outside_window")
        self.assertEqual(late["booking"]["driver"], "Dana Driver")
        none = Scene(booked=False).tools.lookup_booking(PLATE)
        self.assertEqual((none["status"], none["booking"]), ("no_booking", None))

    def test_tools_act_only_on_the_events_plate(self):
        scene = Scene()
        for tool in (scene.tools.lookup_booking, scene.tools.start_session, scene.tools.end_session):
            self.assertIn("error", tool("OTHER99"))
            self.assertIn("error", tool("test-123x"))
        self.assertEqual(scene.db.sessions(), [])
        # The same plate written differently is still the event's plate.
        self.assertEqual(scene.tools.lookup_booking("test 123")["status"], "active")

    def test_one_open_session_per_plate(self):
        scene = Scene()
        first = scene.tools.start_session(PLATE)
        self.assertEqual((first["session_id"], first["status"], first["driver"]), (1, "open", "Dana Driver"))
        second = scene.tools.start_session(PLATE)
        self.assertIn("error", second)
        self.assertIn("session_id 1", second["error"])
        self.assertEqual(len(scene.db.sessions()), 1)

    def test_a_stay_needs_an_active_booking(self):
        self.assertIn("error", Scene(booked=False).tools.start_session(PLATE))
        self.assertIn("error", Scene(at=T0 + timedelta(hours=20)).tools.start_session(PLATE))

    def test_end_session_needs_an_open_stay(self):
        scene = Scene()
        self.assertIn("error", scene.tools.end_session(PLATE))
        scene.tools.start_session(PLATE)
        ended = scene.rebind("LEFT", at=T0 + timedelta(seconds=90)).end_session(PLATE)
        self.assertEqual((ended["session_id"], ended["status"]), (1, "closed"))
        self.assertEqual(ended["left_at"], "2026-01-01T12:01:30+00:00")
        self.assertIn("error", scene.tools.end_session(PLATE))

    def test_compute_bill_rules(self):
        scene = Scene()
        self.assertIn("error", scene.tools.compute_bill(1))  # no such stay
        self.assertIn("error", scene.tools.compute_bill("one"))
        scene.tools.start_session(PLATE)
        self.assertIn("error", scene.tools.compute_bill(1))  # still open
        scene.rebind("LEFT", at=T0 + timedelta(seconds=19.6)).end_session(PLATE)
        bill = scene.tools.compute_bill(1)
        self.assertEqual((bill["minutes"], bill["amount_cents"], bill["amount"]), (1, 9, "$0.09"))
        self.assertEqual(scene.db.session(1)["amount_cents"], 9)
        # A stay of another plate cannot be billed from this event.
        other = scene.db.add_booking("OTHER99", "Ola", T0 - timedelta(hours=1), T0 + timedelta(hours=1), RATE)
        other_session = scene.db.start_session("OTHER99", other, T0)
        scene.db.close_session(other_session, T0 + timedelta(minutes=2))
        self.assertIn("error", scene.tools.compute_bill(other_session))

    def test_compute_bill_accepts_a_numeric_string(self):
        scene = Scene()
        scene.tools.start_session(PLATE)
        scene.rebind("LEFT", at=T0 + timedelta(minutes=61)).end_session(PLATE)
        self.assertEqual(scene.tools.compute_bill("1")["amount_cents"], 509)

    def test_set_light(self):
        scene = Scene()
        self.assertIn("error", scene.tools.set_light("blue"))
        self.assertEqual(scene.light.history, [])
        result = scene.tools.set_light("GREEN")
        self.assertEqual(result["light"], "green")
        self.assertIsInstance(result["seconds_after_event"], float)
        scene.tools.set_light("off")
        self.assertEqual(scene.light.history, ["green", "off"])

    def test_send_alert_attaches_the_snapshot_and_knows_the_driver(self):
        scene = Scene()
        self.assertIn("error", scene.tools.send_alert("police", "hello"))
        self.assertIn("error", scene.tools.send_alert("owner", "   "))
        self.assertEqual(scene.alerts.sent, [])
        owner = scene.tools.send_alert("owner", "Unknown car in the driveway.")
        self.assertEqual(owner, {"sent_to": "owner", "name": None, "snapshot": SNAPSHOT})
        driver = scene.tools.send_alert("driver", "Welcome.")
        self.assertEqual(driver["name"], "Dana Driver")
        self.assertEqual([(a.to, a.name, a.snapshot) for a in scene.alerts.sent],
                         [("owner", None, SNAPSHOT), ("driver", "Dana Driver", SNAPSHOT)])

    def test_no_driver_to_alert_without_a_booking(self):
        self.assertIn("error", Scene(booked=False).tools.send_alert("driver", "Welcome."))

    def test_call_never_raises(self):
        scene = Scene()
        self.assertIn("error", scene.tools.call("open_gate", "{}"))
        self.assertIn("error", scene.tools.call("set_light", "{not json"))
        self.assertIn("error", scene.tools.call("set_light", "[1, 2]"))
        self.assertIn("error", scene.tools.call("set_light", "{}"))
        self.assertIn("error", scene.tools.call("set_light", '{"color": "green", "blink": true}'))
        self.assertEqual(scene.light.history, [])
        self.assertEqual(scene.tools.call("set_light", '{"color": "red"}')["light"], "red")
        self.assertEqual(scene.tools.call("lookup_booking", '{"plate": "TEST123"}')["status"], "active")


if __name__ == "__main__":
    unittest.main()
