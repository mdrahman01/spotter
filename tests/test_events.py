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

class OneCarOnePlateTest(unittest.TestCase):
    """While a plate is present, a read within two edits of it is the same car misread."""

    def arrived(self, rule, plate="MAYR473", at=0.0):
        events = []
        for i in range(5):
            events += rule.update(at + i * 0.5, {plate})
        self.assertEqual([e.type for e in events], ["ARRIVED"])
        return events[0]

    def test_a_misread_counts_as_a_sighting_of_the_present_plate(self):
        from spotter.events import edit_distance
        self.assertEqual(edit_distance("MAYR473", "MAYR478"), 1)
        self.assertEqual(edit_distance("MAYR473", "MAYP47"), 2)  # one substitution, one deletion
        self.assertEqual(edit_distance("MAYR473", "ABC123"), 3)  # capped: more than two
        rule = ArriveLeaveRule(arrive_reads=5, arrive_window_s=3, leave_after_s=4)
        self.arrived(rule)
        for i in range(6):  # six frames of a misread: would have been a second arrival
            events = rule.update(3 + i * 0.5, {"MAYR478"})
            self.assertEqual(events, [])
        self.assertEqual(set(rule.present), {"MAYR473"})
        self.assertEqual(rule.present["MAYR473"].last_seen, 5.5)
        self.assertEqual(rule.merged, 6)
        self.assertEqual(rule.resolve({"MAYR478", "MAYR4T3", "ABC123"}, count=False), {"MAYR473", "ABC123"})
        gone = rule.update(20, set())  # unread since 5.5: the next update declares it gone, last seen at the misread
        self.assertEqual([(e.type, e.last_seen) for e in gone], [("LEFT", 5.5)])
        self.assertEqual(rule.present, {})

    def test_a_different_plate_still_arrives(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)
        events = []
        for i in range(5):
            events += rule.update(3 + i * 0.5, {"MAYR473", "ABC123"})
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "ABC123")])
        self.assertEqual(rule.merged, 0)

    def test_a_far_misread_never_seen_beside_the_present_plate_is_the_same_car(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)  # MAYR473 present
        # the reader then returns a three-edit misread in frames where the real text is not read
        events = []
        for i in range(8):
            events += rule.update(3 + i * 0.5, {"AXR477"})
        self.assertEqual(events, [])  # no second car
        self.assertEqual(set(rule.present), {"MAYR473"})
        self.assertEqual(rule.present["MAYR473"].last_seen, 6.5)  # those reads were sightings of the car
        self.assertEqual(rule.merged, 8)  # five at the judgement, then every later read of the alias
        self.assertEqual(rule._recent, {})
        self.assertEqual(rule.resolve({"AXR477"}, count=False), {"MAYR473"})
        gone = rule.update(11, set())
        self.assertEqual([e.type for e in gone], ["LEFT"])
        self.assertEqual(rule._aliases, {})  # the alias goes with the car

    def test_a_second_plate_seen_beside_the_first_arrives(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)
        events = []
        for i in range(5):
            events += rule.update(3 + i * 0.5, {"MAYR473", "ABC123"})  # two plates in the same frames: two cars
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "ABC123")])
        self.assertEqual(rule.merged, 0)
        # a third text read on its own while both are present is a misread of the nearest one
        for i in range(5):
            events += rule.update(6 + i * 0.5, {"ABC128"})
        self.assertEqual(len(events), 1)
        self.assertEqual(rule.present["ABC123"].last_seen, 8.0)

    def test_a_near_text_read_beside_the_present_plate_is_a_second_car(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)
        events = []
        for i in range(5):
            events += rule.update(3 + i * 0.5, {"MAYR473", "AXR477"})  # three edits apart, but read in the same frames
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "AXR477")])
        self.assertEqual(rule.merged, 0)
        self.assertEqual(set(rule.present), {"MAYR473", "AXR477"})

    def test_a_near_text_read_beside_the_present_plate_once_then_alone_still_arrives(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)
        events = rule.update(3, {"MAYR473", "AXR477"})  # once in the same frame as the present plate
        for i in range(1, 5):
            events += rule.update(3 + i * 0.5, {"AXR477"})
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "AXR477")])
        self.assertEqual(rule.merged, 0)

    def test_a_completely_different_plate_arrives_even_if_never_seen_beside_the_first(self):
        rule = ArriveLeaveRule(5, 3, 4)
        self.arrived(rule)  # MAYR473 present; its plate is then hidden and a different car's plate is read alone
        events = []
        for i in range(5):
            events += rule.update(3 + i * 0.5, {"ABC123"})
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "ABC123")])

    def test_with_no_car_present_a_new_plate_arrives_as_before(self):
        rule = ArriveLeaveRule(5, 3, 4)
        events = []
        for i in range(5):
            events += rule.update(i * 0.5, {"AXR477"})
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "AXR477")])

    def test_at_arrival_the_text_read_most_often_wins(self):
        rule = ArriveLeaveRule(5, 3, 4)
        events = []
        # the reader wavers: six reads of the right text, five of a one-character misread
        reads = ["MAYR473", "MAYR478", "MAYR473", "MAYR478", "MAYR473", "MAYR478", "MAYR473", "MAYR478", "MAYR473", "MAYR478", "MAYR473"]
        for i, text in enumerate(reads):
            events += rule.update(i * 0.25, {text})
            if events:
                break
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "MAYR473")])
        self.assertEqual(events[0].reads, 9)  # the pooled reads in the window
        self.assertEqual(events[0].first_seen, 0.0)
        self.assertEqual(rule.merged, 4)
        self.assertEqual(rule._recent, {})
        # the misread wins when it was read more often
        rule = ArriveLeaveRule(5, 3, 4)
        events = []
        for i, text in enumerate(["MAYR478"] * 6 + ["MAYR473"] * 5):
            events += rule.update(i * 0.25, {text})
        self.assertEqual([(e.type, e.plate) for e in events], [("ARRIVED", "MAYR478")])


if __name__ == "__main__":
    unittest.main()
