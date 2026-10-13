"""Tests of the scene module with no network and no endpoint: the report parser, the Cosmos
client against a faked server, scripted reports, the worker, the two-look rule, and every
branch of the departure (use 1) and unreadable-vehicle (use 2) logic."""

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from spotter import config
from spotter.events import ArriveLeaveRule
from spotter.scene import (
    SPOT_CLEAR,
    UNREADABLE_VEHICLE,
    Agreement,
    CosmosScene,
    Look,
    Report,
    SceneSettings,
    SceneWatch,
    SceneWorker,
    ScriptedScene,
    make_scene,
    parse_report,
    question,
)

FRAME = np.full((72, 128, 3), 90, dtype=np.uint8)
PLATE = "TEST123"


def quiet(line: str) -> None:
    pass


class ReportTest(unittest.TestCase):
    def test_valid_json_becomes_a_report_and_what_is_cut_to_twelve_words(self):
        report = parse_report(json.dumps({"vehicle_in_driveway": True, "plate_visible": False, "what": " ".join(f"w{i}" for i in range(20))}), 1.5)
        self.assertEqual((report.vehicle_in_driveway, report.plate_visible, report.latency_s), (True, False, 1.5))
        self.assertEqual(len(report.what.split()), 12)

    def test_anything_else_is_no_report(self):
        for text in ("", "yes", "{}", '{"vehicle_in_driveway": "true", "plate_visible": false, "what": "x"}',
                     '{"vehicle_in_driveway": true, "plate_visible": false}', "[1]"):
            self.assertIsNone(parse_report(text), text)

    def test_the_question_names_the_spot_and_excludes_the_street(self):
        text = question("the paved driveway in the foreground")
        self.assertIn("the paved driveway in the foreground", text)
        self.assertIn("ignore any vehicle on the street", text)
        self.assertIn("twelve words", text)


class CosmosSceneTest(unittest.TestCase):
    def scene(self, answer):
        calls = []

        def post(url, body, content_type, headers, timeout_s):
            calls.append((url, json.loads(body), headers, timeout_s))
            if isinstance(answer, Exception):
                raise answer
            return {"choices": [{"message": {"content": answer}}]}

        return CosmosScene("https://cosmos.example/", "tok-secret", "the driveway", timeout_s=15, post=post, say=quiet), calls

    def test_a_look_sends_the_frame_as_a_data_url_with_structured_output(self):
        scene, calls = self.scene(json.dumps({"vehicle_in_driveway": True, "plate_visible": True, "what": "a white SUV"}))
        report = scene.look(FRAME)
        self.assertEqual((report.vehicle_in_driveway, report.plate_visible, report.what), (True, True, "a white SUV"))
        url, body, headers, timeout_s = calls[0]
        self.assertEqual(url, "https://cosmos.example/v1/chat/completions")
        self.assertEqual(timeout_s, 15)
        self.assertEqual(headers["Authorization"], "Bearer tok-secret")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["required"], ["vehicle_in_driveway", "plate_visible", "what"])
        content = body["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertIn("the driveway", content[1]["text"])

    def test_a_failed_or_malformed_look_is_unavailable_and_the_token_is_not_in_the_error(self):
        scene, _ = self.scene(TimeoutError("timed out for tok-secret"))
        self.assertIsNone(scene.look(FRAME))
        self.assertIn("TimeoutError", scene.last_error)
        self.assertNotIn("tok-secret", scene.last_error)
        scene, _ = self.scene("I think there is a car.")
        self.assertIsNone(scene.look(FRAME))
        self.assertIn("not the JSON asked for", scene.last_error)


class ScriptedSceneTest(unittest.TestCase):
    def test_entries_apply_by_time_and_unavailable_answers_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.json"
            path.write_text(json.dumps([
                {"until": 10, "vehicle_in_driveway": False, "plate_visible": False, "what": "empty asphalt"},
                {"until": 20, "unavailable": True},
                {"vehicle_in_driveway": True, "plate_visible": True, "what": "a white SUV"},
            ]))
            scene = ScriptedScene(path)
            self.assertFalse(scene.look(FRAME, 5).vehicle_in_driveway)
            self.assertIsNone(scene.look(FRAME, 15))
            self.assertTrue(scene.look(FRAME, 25).vehicle_in_driveway)
            self.assertTrue(scene.look(FRAME, 999).vehicle_in_driveway)
            self.assertIsInstance(make_scene(f"script:{path}", {}, ""), ScriptedScene)

    def test_settings(self):
        self.assertEqual(config.load({}, env={}).scene, "off")
        self.assertIsNone(make_scene("off", {}, ""))
        with self.assertRaises(ValueError):
            config.load({}, env={"SPOTTER_SCENE": "maybe"})
        with self.assertRaises(ValueError):
            make_scene("cosmos", {}, "the driveway")  # no URL or token
        with self.assertRaises(ValueError):
            make_scene("cosmos", {"COSMOS_URL": "https://x", "COSMOS_TOKEN": "t"}, "")  # no spot description
        self.assertIsInstance(make_scene("cosmos", {"COSMOS_URL": "https://x", "COSMOS_TOKEN": "t"}, "the driveway"), CosmosScene)
        settings = config.load({}, env={"SPOTTER_SCENE": "cosmos", "SPOTTER_SCENE_CHECK_AFTER_S": "5"})
        self.assertEqual((settings.scene, settings.scene_check_after_s, settings.scene_idle_every_s), ("cosmos", 5, 15))


class FakeScene:
    """Answers from a list of reports (None = unavailable), repeating the last one."""

    def __init__(self, reports):
        self.reports = list(reports)
        self.looks = 0
        self.last_error = "faked"

    def look(self, frame):
        self.looks += 1
        report = self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]
        return report


def car(what="a white SUV", plate=True):
    return Report(True, plate, what)


def empty():
    return Report(False, False, "empty asphalt")


class WorkerTest(unittest.TestCase):
    def test_one_request_in_flight_results_come_back_tagged(self):
        worker = SceneWorker(FakeScene([car(), empty()]), say=quiet)
        try:
            self.assertTrue(worker.request(FRAME, 1.0, "idle"))
            self.assertFalse(worker.request(FRAME, 1.1, "idle"))  # one at a time
            deadline = time.time() + 2
            looks = []
            while not looks and time.time() < deadline:
                looks = worker.results()
                time.sleep(0.01)
            self.assertEqual((looks[0].tag, looks[0].asked_at, looks[0].report.vehicle_in_driveway), ("idle", 1.0, True))
            self.assertEqual(worker.looks, 1)
        finally:
            worker.stop()


class AgreementTest(unittest.TestCase):
    def test_two_in_a_row_that_agree_and_unavailable_resets(self):
        a = Agreement()
        a.add(Look("idle", 1, car()))
        self.assertFalse(a.agreed(True))
        a.add(Look("idle", 2, car()))
        self.assertTrue(a.agreed(True))
        a.add(Look("idle", 3, empty()))
        self.assertFalse(a.agreed(True))
        self.assertFalse(a.agreed(False))
        a.add(Look("idle", 4, None))
        a.add(Look("idle", 5, empty()))
        self.assertFalse(a.agreed(False))  # the unavailable look broke the run
        a.add(Look("idle", 6, empty()))
        self.assertTrue(a.agreed(False))
        self.assertEqual(a.first_at, 5)


class Driveway:
    """Drives SceneWatch frame by frame with an instant, scripted scene (looks finish before the next tick)."""

    def __init__(self, reports, leave_after_s=4, check_after_s=2, relook_every_s=3, idle_every_s=2, unreadable_after_s=4):
        self.rule = ArriveLeaveRule(arrive_reads=2, arrive_window_s=3, leave_after_s=leave_after_s)
        self.scene = FakeScene(reports)
        self.worker = SceneWorker(self.scene, say=quiet)
        self.watch = SceneWatch(self.worker, self.rule, SceneSettings(check_after_s, relook_every_s, idle_every_s, unreadable_after_s), say=quiet)
        self.last_read = {}
        self.events = []

    def frame(self, now, plates=()):
        plates = set(plates)
        for event in self.rule.update(now, plates):
            self.events.append((event.type, event.plate, now, event.note))
        for plate in plates:
            if plate in self.rule.present:
                self.last_read[plate] = now
        decisions = self.watch.tick(now, FRAME, plates, {p: self.last_read.get(p, now) for p in self.rule.present})
        # let the worker finish so the next tick sees the result
        deadline = time.time() + 1
        while self.worker.pending and time.time() < deadline:
            time.sleep(0.005)
        for d in decisions:
            if d.kind == "LEFT_CONFIRMED":
                gone = self.rule.leave_now(d.plate, now, "confirmed by scene")
                self.events.append(("LEFT", d.plate, now, gone.note))
            else:
                self.events.append((d.kind, d.what, now, (d.since, d.until)))
        return decisions

    def run(self, until, plates_at, step=0.5):
        now = 0.0
        while now <= until + 1e-9:
            self.frame(now, plates_at(now))
            now = round(now + step, 6)

    def close(self):
        self.worker.stop()


class DepartureTest(unittest.TestCase):
    """Use 1: a present plate unread for check_after_s gets looks."""

    def test_two_looks_with_no_vehicle_confirm_the_departure_early(self):
        d = Driveway([empty()])
        try:
            d.run(12, lambda now: {PLATE} if now <= 4 else set())
            lefts = [e for e in d.events if e[0] == "LEFT"]
            self.assertEqual(len(lefts), 1)
            self.assertEqual(lefts[0][3], "confirmed by scene")
            self.assertLess(lefts[0][2], 8)  # the plain rule would have fired at 4 + 4 = 8
            self.assertGreaterEqual(d.scene.looks, 2)  # two departure looks, then idle looks while the spot is empty
        finally:
            d.close()

    def test_two_looks_seeing_the_car_hold_the_stay_until_the_scene_says_it_has_gone(self):
        # The car is seen by the scene until rule time 12, then gone; the plate is never read again.
        reports = [car()] * 4 + [empty()]
        d = Driveway(reports, relook_every_s=3)
        try:
            d.run(20, lambda now: {PLATE} if now <= 4 else set())
            lefts = [e for e in d.events if e[0] == "LEFT"]
            self.assertEqual(len(lefts), 1)
            self.assertGreater(lefts[0][2], 10)  # held well past 4 + leave_after_s
            self.assertEqual(lefts[0][3], "confirmed by scene")
        finally:
            d.close()

    def test_the_plate_read_again_releases_the_hold(self):
        d = Driveway([car()])
        try:
            d.run(9, lambda now: {PLATE} if now <= 2 or 8 <= now else set())
            self.assertIn(PLATE, d.rule.present)
            self.assertNotIn(PLATE, d.rule.held)
            self.assertEqual([e for e in d.events if e[0] == "LEFT"], [])
        finally:
            d.close()

    def test_an_unavailable_scene_leaves_the_plain_rule_in_charge(self):
        d = Driveway([car(), car(), None])  # held, then the scene stops answering
        try:
            d.run(14, lambda now: {PLATE} if now <= 2 else set())
            lefts = [e for e in d.events if e[0] == "LEFT"]
            self.assertEqual(len(lefts), 1)
            self.assertEqual(lefts[0][3], "")  # by the leave_after_s rule, not the scene
        finally:
            d.close()

    def test_another_plate_arriving_means_the_held_car_has_left(self):
        d = Driveway([car()])
        try:
            d.run(10, lambda now: {PLATE} if now <= 2 else ({"OTHER77"} if now >= 8 else set()))
            kinds = [(e[0], e[1]) for e in d.events]
            self.assertEqual(kinds, [("ARRIVED", PLATE), ("LEFT", PLATE), ("ARRIVED", "OTHER77")])
            self.assertEqual([e[3] for e in d.events if e[0] == "LEFT"], ["another plate arrived"])
        finally:
            d.close()


class UnreadableVehicleTest(unittest.TestCase):
    """Use 2: looks while no car is present."""

    def test_a_vehicle_seen_long_enough_with_no_plate_is_raised_once_then_cleared(self):
        # looks every 2 s: empty at 0, vehicle from 2 to 12, empty after
        reports = [empty(), car("a dark van", plate=False)] + [car("a dark van", plate=False)] * 5 + [empty()]
        d = Driveway(reports, idle_every_s=2, unreadable_after_s=4)
        try:
            d.run(26, lambda now: set())
            kinds = [e[0] for e in d.events]
            self.assertEqual(kinds, [UNREADABLE_VEHICLE, SPOT_CLEAR])
            unreadable, clear = d.events
            self.assertEqual(unreadable[1], "a dark van")
            self.assertGreaterEqual(unreadable[2], 6)  # first seen at about 2.5, raised 4 s later
            since, until = clear[3]
            self.assertTrue(2 <= since <= 3.5, since)
            self.assertGreaterEqual(until - since, 8)
            self.assertFalse(d.watch.unreadable_raised)
        finally:
            d.close()

    def test_a_plate_read_lets_the_normal_arrival_take_over(self):
        d = Driveway([car(plate=False)], idle_every_s=2, unreadable_after_s=4)
        try:
            d.run(16, lambda now: {PLATE} if now >= 9 else set())
            kinds = [e[0] for e in d.events]
            self.assertEqual(kinds, [UNREADABLE_VEHICLE, "ARRIVED"])  # no SPOT_CLEAR: the known car is here
            self.assertIsNone(d.watch.vehicle_since)
        finally:
            d.close()

    def test_one_look_decides_nothing_and_unavailable_looks_do_not_count(self):
        reports = [car(plate=False), None, car(plate=False), None, car(plate=False), None]
        d = Driveway(reports, idle_every_s=2, unreadable_after_s=4)
        try:
            d.run(14, lambda now: set())
            self.assertEqual(d.events, [])
            self.assertGreaterEqual(d.scene.looks, 5)
        finally:
            d.close()


class RuleHoldTest(unittest.TestCase):
    def test_hold_release_and_leave_now(self):
        rule = ArriveLeaveRule(2, 3, 4)
        rule.update(0, {PLATE}); rule.update(1, {PLATE})
        self.assertIn(PLATE, rule.present)
        rule.hold(PLATE)
        self.assertEqual(rule.update(10, set()), [])  # held: no LEFT at 1 + 4
        rule.release(PLATE)
        events = rule.update(10.5, set())
        self.assertEqual([(e.type, e.last_seen) for e in events], [("LEFT", 1)])
        rule.update(11, {PLATE}); rule.update(12, {PLATE})
        gone = rule.leave_now(PLATE, 13, "confirmed by scene")
        self.assertEqual((gone.type, gone.last_seen, gone.note), ("LEFT", 12, "confirmed by scene"))
        self.assertIsNone(rule.leave_now(PLATE, 14))


if __name__ == "__main__":
    unittest.main()
