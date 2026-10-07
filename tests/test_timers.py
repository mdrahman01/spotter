"""Tests of the timer rule: ENDING_SOON and OVERSTAY, once per stay, on an in-memory database."""

import unittest
from datetime import datetime, timedelta, timezone

from spotter.db import Database
from spotter.timers import ENDING_SOON, OVERSTAY, StayTimer, due

PLATE = "TEST123"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
ENDS = T0 + timedelta(seconds=12)


def seconds(n: float) -> datetime:
    return T0 + timedelta(seconds=n)


class Scene:
    """A booking that ends 12 s after T0, with ending_soon_s 4 and overstay_grace_s 3."""

    def __init__(self, arrived_at=T0):
        self.db = Database(":memory:")
        self.booking_id = self.db.add_booking(PLATE, "Dana Driver", T0 - timedelta(minutes=1), ENDS, 500)
        self.session_id = self.db.start_session(PLATE, self.booking_id, arrived_at)
        self.timer = StayTimer(self.db, ending_soon_s=4, overstay_grace_s=3)

    def kinds(self, at: float) -> list[str]:
        return [event.type for event in self.timer.check(seconds(at))]


class TimerRuleTest(unittest.TestCase):
    def test_ending_soon_fires_once_when_the_booking_ends_within_the_window(self):
        scene = Scene()
        self.assertEqual(scene.kinds(7), [])
        self.assertEqual(scene.kinds(7.9), [])
        self.assertEqual(scene.kinds(8), [ENDING_SOON])
        self.assertEqual(scene.kinds(9), [])
        self.assertEqual(scene.kinds(11.9), [])
        self.assertEqual(scene.db.session(scene.session_id)["ending_soon_at"], "2026-01-01T12:00:08+00:00")
        self.assertIsNone(scene.db.session(scene.session_id)["overstayed_at"])

    def test_overstay_fires_once_after_the_grace_and_marks_the_stay(self):
        scene = Scene()
        scene.kinds(8)
        self.assertEqual(scene.kinds(12), [])
        self.assertEqual(scene.kinds(14.9), [])
        self.assertEqual(scene.kinds(15), [OVERSTAY])
        self.assertEqual(scene.kinds(16), [])
        self.assertEqual(scene.kinds(60), [])
        self.assertEqual(scene.db.session(scene.session_id)["overstayed_at"], "2026-01-01T12:00:15+00:00")

    def test_events_carry_the_plate_the_stay_and_the_booking_end(self):
        scene = Scene()
        event = scene.timer.check(seconds(8))[0]
        self.assertEqual((event.type, event.plate, event.session_id, event.booking_ends_at),
                         (ENDING_SOON, PLATE, scene.session_id, ENDS))

    def test_a_car_arriving_inside_the_window_is_warned_at_once(self):
        scene = Scene(arrived_at=seconds(10))
        self.assertEqual(scene.kinds(10), [ENDING_SOON])

    def test_no_warning_after_the_booking_has_ended(self):
        scene = Scene()
        self.assertEqual(scene.kinds(13), [])  # too late to warn, too early for overstay
        self.assertEqual(scene.kinds(15), [OVERSTAY])
        self.assertIsNone(scene.db.session(scene.session_id)["ending_soon_at"])

    def test_closed_stays_are_left_alone(self):
        scene = Scene()
        scene.db.close_session(scene.session_id, seconds(11))
        self.assertEqual(scene.kinds(8), [])
        self.assertEqual(scene.kinds(30), [])

    def test_each_open_stay_is_checked(self):
        scene = Scene()
        other = scene.db.add_booking("OTHER99", "Ola", T0, T0 + timedelta(hours=1), 500)
        scene.db.start_session("OTHER99", other, T0)
        events = scene.timer.check(seconds(15))
        self.assertEqual([(e.type, e.plate) for e in events], [(OVERSTAY, PLATE)])

    def test_due_is_pure(self):
        booking = {"ends_at": "2026-01-01T12:00:12+00:00"}
        fresh = {"ending_soon_at": None, "overstayed_at": None}
        self.assertEqual(due(fresh, booking, seconds(8), 4, 3), [ENDING_SOON])
        self.assertEqual(due(fresh, booking, seconds(12), 4, 3), [ENDING_SOON])
        self.assertEqual(due(fresh, booking, seconds(13), 4, 3), [])
        self.assertEqual(due(fresh, booking, seconds(15), 4, 3), [OVERSTAY])
        warned = {"ending_soon_at": "x", "overstayed_at": None}
        self.assertEqual(due(warned, booking, seconds(9), 4, 3), [])
        self.assertEqual(due({"ending_soon_at": "x", "overstayed_at": "y"}, booking, seconds(99), 4, 3), [])


if __name__ == "__main__":
    unittest.main()
