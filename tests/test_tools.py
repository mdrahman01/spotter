"""Tests of the tool rules and the bill arithmetic: in-memory database, no network, no model."""

import unittest
from datetime import datetime, timedelta, timezone

from spotter.alerts import Alert, ConsoleAlerts
from spotter.db import Database
from spotter.light import ConsoleLight
from spotter.tools import EventContext, Toolbox, bill_cents, public
from spotter.words import WordBook, clock_words, duration_words

PLATE = "TEST123"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
SNAPSHOT = "data/snapshots/test.jpg"
RATE = 500  # cents per hour


def quiet(line: str) -> None:
    pass


class FailingAlerts:
    """An alerts backend whose every send fails, as Telegram would without a network."""

    def send(self, alert: Alert) -> str | None:
        return "URLError: simulated outage"


class Scene:
    """A database, the stand-ins and a toolbox bound to one event."""

    def __init__(self, event_type="ARRIVED", plate=PLATE, at=T0, booked=True, min_charge=0, alerts=None,
                 overstay_fee=500):
        self.db = Database(":memory:")
        if booked:
            self.booking_id = self.db.add_booking(
                PLATE, "Dana Driver", T0 - timedelta(hours=1), T0 + timedelta(hours=11), RATE
            )
        self.light = ConsoleLight(say=quiet)
        self.alerts = alerts or ConsoleAlerts(say=quiet)
        self.min_charge = min_charge
        self.overstay_fee = overstay_fee
        self.tools = self.rebind(event_type, plate, at)

    def rebind(self, event_type, plate=PLATE, at=T0, **frames) -> Toolbox:
        """A toolbox for a later event on the same database; frames may give last_read_at etc."""
        context = EventContext(event_type, plate, at, SNAPSHOT, **frames)
        self.tools = Toolbox(self.db, self.light, self.alerts, context, self.min_charge, self.overstay_fee)
        return self.tools


class BillArithmeticTest(unittest.TestCase):
    def bill(self, seconds, rate=RATE, min_charge=0):
        return bill_cents(T0, T0 + timedelta(seconds=seconds), rate, min_charge)

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

    def test_minimum_charge(self):
        self.assertEqual(self.bill(10, min_charge=100), (1, 100))  # 9 cents lifted to the minimum
        self.assertEqual(self.bill(60 * 11, min_charge=100), (11, 100))  # 92 cents lifted
        self.assertEqual(self.bill(60 * 13, min_charge=100), (13, 109))  # above it: untouched
        self.assertEqual(self.bill(7200, min_charge=100), (120, 1000))
        self.assertEqual(self.bill(60, rate=0, min_charge=100), (1, 100))

    def test_clock_going_backwards_is_billed_as_one_minute(self):
        self.assertEqual(bill_cents(T0, T0 - timedelta(minutes=5), RATE), (1, 9))


class ToolRulesTest(unittest.TestCase):
    def test_lookup_reports_active_outside_window_and_no_booking(self):
        active = Scene().tools.lookup_booking(PLATE)
        self.assertEqual(active["status"], "active")
        self.assertEqual(active["booking"]["rate"], "$5.00/hour")
        self.assertNotIn("plate", active["booking"])
        late = Scene(at=T0 + timedelta(hours=20)).tools.lookup_booking(PLATE)
        self.assertEqual(late["status"], "outside_window")
        self.assertEqual(late["booking"]["driver"], "Dana Driver")

    def test_an_open_stay_past_its_booking_is_overstaying_not_unbooked(self):
        scene = Scene()
        scene.tools.start_session(PLATE)
        scene.db.mark_overstayed(1, T0 + timedelta(hours=11, minutes=5))
        later = scene.rebind("LEFT", at=T0 + timedelta(hours=12)).lookup_booking(PLATE)
        self.assertEqual(later["status"], "overstaying")
        self.assertEqual(later["booking"]["driver"], "Dana Driver")
        self.assertEqual(later["open_session"]["overstayed"], True)
        none = Scene(booked=False).tools.lookup_booking(PLATE)
        self.assertEqual((none["status"], none["booking"]), ("no_booking", None))

    def test_money_is_shown_in_dollars(self):
        booking = {"id": 1, "driver": "D", "starts_at": "2026-01-01T11:00:00+00:00", "ends_at": "2026-01-01T23:00:00+00:00",
                   "rate_cents_per_hour": 1250, "plate": PLATE}
        shown = public(booking, WordBook(), T0)
        self.assertEqual(shown["rate"], "$12.50/hour")
        self.assertEqual((shown["starts"], shown["ends"]), (clock_words(T0 - timedelta(hours=1), T0), clock_words(T0 + timedelta(hours=11), T0)))
        self.assertNotIn("starts_at", shown)  # times reach the model only as words

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
        open_stay = scene.tools.lookup_booking(PLATE)["open_session"]
        self.assertEqual(open_stay, {"session_id": 1, "arrived": clock_words(T0, T0),
                                     "ending_soon_warned": False, "overstayed": False})
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
        self.assertEqual(scene.db.session(1)["left_at"], "2026-01-01T12:01:30+00:00")
        self.assertEqual((ended["left"], ended["stayed"]), (clock_words(T0 + timedelta(seconds=90), T0), "1 minute 30 seconds"))
        self.assertIn("error", scene.tools.end_session(PLATE))

    def test_compute_bill_rules(self):
        scene = Scene()
        self.assertIn("error", scene.tools.compute_bill(1))  # no such stay
        self.assertIn("error", scene.tools.compute_bill("one"))
        scene.tools.start_session(PLATE)
        self.assertIn("error", scene.tools.compute_bill(1))  # still open
        scene.rebind("LEFT", at=T0 + timedelta(seconds=19.6)).end_session(PLATE)
        bill = scene.tools.compute_bill(1)
        self.assertEqual((bill["time_billed"], bill["amount_cents"], bill["total"]), ("1 minute", 9, "$0.09"))
        self.assertNotIn("minutes", bill)
        self.assertEqual(bill["overstayed"], False)
        self.assertNotIn("overstay_fee", bill)  # no fee applied, so the model sees none
        self.assertNotIn("parking", bill)  # and gets the total only, not parking and total
        self.assertEqual(bill["rate"], "$5.00/hour")
        self.assertNotIn("rate_cents_per_hour", bill)
        self.assertEqual(scene.db.session(1)["amount_cents"], 9)
        # A stay of another plate cannot be billed from this event.
        other = scene.db.add_booking("OTHER99", "Ola", T0 - timedelta(hours=1), T0 + timedelta(hours=1), RATE)
        other_session = scene.db.start_session("OTHER99", other, T0)
        scene.db.close_session(other_session, T0 + timedelta(minutes=2))
        self.assertIn("error", scene.tools.compute_bill(other_session))

    def test_compute_bill_applies_the_minimum_charge(self):
        scene = Scene(min_charge=100)
        scene.tools.start_session(PLATE)
        scene.rebind("LEFT", at=T0 + timedelta(seconds=19.6)).end_session(PLATE)
        bill = scene.tools.compute_bill(1)
        self.assertEqual((bill["amount_cents"], bill["total"]), (100, "$1.00"))
        self.assertTrue(bill["minimum_charge_applied"])
        self.assertNotIn("parking", bill)
        self.assertEqual(scene.db.session(1)["amount_cents"], 100)

    def test_an_overstayed_stay_pays_the_fee_on_top_of_the_minimum_charge(self):
        scene = Scene(min_charge=100)
        scene.tools.start_session(PLATE)
        scene.db.mark_overstayed(1, T0 + timedelta(seconds=15))
        scene.rebind("LEFT", at=T0 + timedelta(seconds=19.6)).end_session(PLATE)
        bill = scene.tools.compute_bill(1)
        self.assertEqual(
            (bill["parking"], bill["overstay_fee"], bill["total"], bill["amount_cents"], bill["overstayed"]),
            ("$1.00", "$5.00", "$6.00", 600, True),
        )
        self.assertEqual(scene.db.session(1)["amount_cents"], 600)

    def test_a_stay_ends_when_the_car_was_last_read(self):
        scene = Scene(min_charge=0)
        scene.tools.start_session(PLATE)
        left = scene.rebind(
            "LEFT",
            at=T0 + timedelta(seconds=26.22),
            last_read_at=T0 + timedelta(seconds=22.22),
            last_read_snapshot="data/snapshots/last-read.jpg",
        )
        ended = left.end_session(PLATE)
        self.assertEqual(ended["left"], clock_words(T0 + timedelta(seconds=22.22), T0))
        session = scene.db.session(1)
        self.assertEqual((session["left_at"], session["left_snapshot"]),
                         ("2026-01-01T12:00:22+00:00", "data/snapshots/last-read.jpg"))
        self.assertEqual(left.compute_bill(1)["time_billed"], "1 minute")
        # LEFT alerts carry the frame of the last sighting, not the frame when LEFT fired.
        self.assertEqual(left.send_alert("driver", "Bye.")["snapshot"], "data/snapshots/last-read.jpg")
        self.assertEqual(scene.alerts.sent[-1].snapshot, "data/snapshots/last-read.jpg")

    def test_other_events_end_and_attach_at_their_own_time(self):
        scene = Scene(min_charge=0)
        scene.tools.start_session(PLATE)
        self.assertEqual(scene.tools.send_alert("owner", "Hi.")["snapshot"], SNAPSHOT)
        scene.rebind("LEFT", at=T0 + timedelta(seconds=90)).end_session(PLATE)
        self.assertEqual(scene.db.session(1)["left_at"], "2026-01-01T12:01:30+00:00")  # no last read given: the event time

    def test_the_overstay_fee_is_a_setting(self):
        scene = Scene(min_charge=0, overstay_fee=250)
        scene.tools.start_session(PLATE)
        scene.db.mark_overstayed(1, T0 + timedelta(hours=2))
        scene.rebind("LEFT", at=T0 + timedelta(hours=2, minutes=1)).end_session(PLATE)
        bill = scene.tools.compute_bill(1)
        self.assertEqual((bill["parking"], bill["overstay_fee"], bill["total"]), ("$10.09", "$2.50", "$12.59"))

    def test_compute_bill_above_the_minimum_is_untouched(self):
        scene = Scene(min_charge=100)
        scene.tools.start_session(PLATE)
        scene.rebind("LEFT", at=T0 + timedelta(minutes=61)).end_session(PLATE)
        bill = scene.tools.compute_bill("1")  # a numeric string is accepted
        self.assertEqual((bill["amount_cents"], bill["minimum_charge_applied"]), (509, False))

    def test_set_light(self):
        scene = Scene()
        self.assertIn("error", scene.tools.set_light("blue"))
        self.assertEqual(scene.light.history, [])
        result = scene.tools.set_light("GREEN")
        self.assertEqual(result["light"], "green")
        self.assertIsInstance(result["seconds_after_event"], float)
        scene.tools.set_light("amber")
        scene.tools.set_light("off")
        self.assertEqual(scene.light.history, ["green", "amber", "off"])
        self.assertEqual(scene.tools.lookup_booking(PLATE)["open_session"], None)

    def test_send_alert_attaches_the_snapshot_and_knows_the_driver(self):
        scene = Scene()
        self.assertIn("error", scene.tools.send_alert("police", "hello"))
        self.assertIn("error", scene.tools.send_alert("owner", "   "))
        self.assertEqual(scene.alerts.sent, [])
        owner = scene.tools.send_alert("owner", "Unknown car in the driveway.")
        self.assertEqual(
            owner,
            {"sent_to": "owner", "name": None, "snapshot": SNAPSHOT, "delivered": True, "delivery_error": None},
        )
        driver = scene.tools.send_alert("driver", "Welcome.")
        self.assertEqual(driver["name"], "Dana Driver")
        self.assertEqual([(a.to, a.name, a.snapshot) for a in scene.alerts.sent],
                         [("owner", None, SNAPSHOT), ("driver", "Dana Driver", SNAPSHOT)])

    def test_no_driver_to_alert_without_a_booking(self):
        self.assertIn("error", Scene(booked=False).tools.send_alert("driver", "Welcome."))

    def test_a_failed_delivery_is_reported_not_raised(self):
        scene = Scene(alerts=FailingAlerts())
        result = scene.tools.send_alert("owner", "Unknown car.")
        self.assertNotIn("error", result)
        self.assertEqual((result["delivered"], result["delivery_error"]), (False, "URLError: simulated outage"))

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
