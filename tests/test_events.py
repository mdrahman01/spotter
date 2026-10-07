"""Pure tests of the arrive/leave rule: fake reads, no video and no model.

    python -m unittest discover -s tests -v
"""

import unittest

from spotter.events import ArriveLeaveRule

PLATE = "TEST123"
STEP = 0.2  # seconds between sampled frames at the default process_fps of 5


def default_rule() -> ArriveLeaveRule:
    """The default settings: 5 reads within 3 s to arrive, 4 s unread to leave."""
    return ArriveLeaveRule(arrive_reads=5, arrive_window_s=3, leave_after_s=4)


def frames(start: float, end: float, plates: list[str]) -> list[tuple[float, list[str]]]:
    """One sampled frame every STEP seconds from start to end inclusive, each reading `plates`."""
    count = round((end - start) / STEP)
    return [(round(start + i * STEP, 6), plates) for i in range(count + 1)]


def run(rule: ArriveLeaveRule, timeline: list[tuple[float, list[str]]]):
    """Feed every frame to the rule; return all events, oldest first."""
    events = []
    for now, plates in timeline:
        events += rule.update(now, plates)
    return events


class ArrivalTest(unittest.TestCase):
    def test_five_reads_within_three_seconds_arrive(self):
        rule = default_rule()
        events = run(rule, frames(0.0, 0.6, [PLATE]))  # 4 reads
        self.assertEqual(events, [])
        events = run(rule, [(0.8, [PLATE])])  # the 5th read
        self.assertEqual(len(events), 1)
        arrived = events[0]
        self.assertEqual((arrived.type, arrived.plate), ("ARRIVED", PLATE))
        self.assertEqual((arrived.time, arrived.first_seen, arrived.reads), (0.8, 0.0, 5))
        self.assertIn(PLATE, rule.present)

    def test_five_reads_spanning_exactly_three_seconds_arrive(self):
        timeline = [(t, [PLATE]) for t in (0.0, 0.75, 1.5, 2.25, 3.0)]
        events = run(default_rule(), timeline)
        self.assertEqual([(e.type, e.time) for e in events], [("ARRIVED", 3.0)])

    def test_four_reads_never_arrive(self):
        timeline = frames(0.0, 0.6, [PLATE]) + frames(0.8, 30.0, [])
        self.assertEqual(run(default_rule(), timeline), [])

    def test_five_reads_spread_over_more_than_three_seconds_never_arrive(self):
        # One read a second: any 3 s window holds at most 4 of them.
        timeline = []
        for t in range(5):
            timeline += [(float(t), [PLATE])] + frames(t + STEP, t + 0.8, [])
        timeline += frames(5.0, 30.0, [])
        self.assertEqual(run(default_rule(), timeline), [])

    def test_a_single_stray_read_never_arrives(self):
        rule = default_rule()
        timeline = frames(0.0, 10.0, [PLATE])
        timeline[25] = (timeline[25][0], [PLATE, "TEST12B"])  # one misread at 5.0 s
        timeline += frames(10.2, 60.0, [])
        events = run(rule, timeline)
        self.assertEqual(
            [(e.type, e.plate, e.time) for e in events],
            [("ARRIVED", PLATE, 0.8), ("LEFT", PLATE, 14.0)],
        )

    def test_a_plate_read_twice_in_one_frame_counts_once(self):
        timeline = frames(0.0, 0.4, [PLATE, PLATE]) + frames(0.6, 10.0, [])
        self.assertEqual(run(default_rule(), timeline), [])


class LeavingTest(unittest.TestCase):
    def test_a_gap_shorter_than_leave_after_does_not_end_the_stay(self):
        rule = default_rule()
        timeline = (
            frames(0.0, 2.0, [PLATE])
            + frames(2.2, 5.8, [])  # unread for 3.8 s
            + frames(6.0, 10.0, [PLATE])
        )
        events = run(rule, timeline)
        self.assertEqual([(e.type, e.time) for e in events], [("ARRIVED", 0.8)])
        self.assertIn(PLATE, rule.present)
        self.assertEqual(rule.present[PLATE].first_seen, 0.0)

    def test_a_longer_gap_ends_the_stay(self):
        rule = default_rule()
        timeline = frames(0.0, 2.0, [PLATE]) + frames(2.2, 10.0, [])
        events = run(rule, timeline)
        self.assertEqual([(e.type, e.time) for e in events], [("ARRIVED", 0.8), ("LEFT", 6.0)])
        left = events[1]
        self.assertEqual((left.first_seen, left.last_seen, left.reads), (0.0, 2.0, 11))
        self.assertEqual(rule.present, {})


if __name__ == "__main__":
    unittest.main()
