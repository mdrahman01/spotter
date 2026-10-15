"""Tests of the dashboard's JSON and frame endpoints with no network: a file database, a
LiveState fed by hand, and the server on an ephemeral port on localhost."""

import json
import tempfile
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from spotter import config, services
from spotter.dashboard import Dashboard, LiveState
from spotter.db import Database
from spotter.light import ConsoleLight

NOW = datetime(2026, 10, 14, 0, 30, tzinfo=timezone.utc)
PLATE = "MAYR473"


def quiet(line: str) -> None:
    pass


class Scene:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "spotter.db"
        db = Database(self.db_path)
        booking = db.add_booking(PLATE, "Dana Driver", NOW - timedelta(hours=1), NOW + timedelta(minutes=10), 500)
        self.session = db.start_session(PLATE, booking, NOW - timedelta(minutes=25), "data/snapshots/a.jpg")
        db.mark_overstayed(self.session, NOW - timedelta(minutes=2), NOW - timedelta(minutes=3))
        event = {"type": "ARRIVED", "plate": PLATE, "time": (NOW - timedelta(minutes=25)).isoformat(timespec="seconds")}
        db.log_action(event, "lookup_booking", {"plate": PLATE}, {"status": "active"})
        db.log_action(event, "set_light", {"color": "green"}, {"light": "green", "seconds_after_event": 1.2})
        db.log_action(event, "send_alert", {"to": "driver", "message": "Welcome."}, {"sent_to": "driver", "delivered": True})
        db.log_action(event, "final", {"round": 3}, {"text": "Started the stay, lit green, alerted the driver."})
        db.log_action(event, "words", {"supplied": ["11:05 pm"]}, {"count": 1})
        db.log_action(event, "finish", {"how": "clean"}, {"reminded_about": [], "completed_by_code": []})
        camera = {"type": "CAMERA_OFFLINE", "time": (NOW - timedelta(minutes=10)).isoformat(timespec="seconds")}
        db.log_action(camera, "camera", {"downtime_s": None}, {"by": "code"})
        db.log_action(camera, "send_alert", {"to": "owner", "message": "Camera offline.", "by": "code"}, {"sent_to": "owner", "delivered": True, "by": "code"})
        price = {"type": "PRICE_CHECK", "time": (NOW - timedelta(minutes=5)).isoformat(timespec="seconds")}
        db.log_action(price, "search_events", {"query": "events"}, {"results": [{"title": "Big Arena concert", "url": "https://example.com/a"}], "searches_left": 2})
        db.log_action(price, "set_demand", {"level": "busy", "event": "Concert", "reason": "busy", "source_url": "https://example.com/a"}, {"level": "busy", "price": "$8.00 an hour"})
        db.log_action(price, "finish", {"how": "clean"}, {"level": "busy", "source_url": "https://example.com/a"})
        db.set_price("busy", 800, NOW - timedelta(minutes=5), NOW + timedelta(hours=6), "Concert", "busy", "https://example.com/a")
        db.close()
        self.settings = config.load({}, env={"SPOTTER_LOT_NAME": "Elm Street lot", "SPOTTER_DASHBOARD_PORT": "0"})
        self.live = LiveState()
        self.light = ConsoleLight(say=quiet)
        self.light.set("amber")
        self.dashboard = Dashboard(self.settings, self.live, self.light, db_path=self.db_path, now=lambda: NOW)

    def close(self):
        self.tmp.cleanup()


class StateTest(unittest.TestCase):
    def setUp(self):
        services.reset()
        self.scene = Scene()

    def tearDown(self):
        self.scene.close()

    def test_the_state_has_everything_the_page_shows(self):
        scene = self.scene
        scene.live.set_camera("live", NOW - timedelta(minutes=9))
        scene.live.set_read(PLATE, NOW - timedelta(seconds=4))
        scene.live.set_scene(True, {"vehicle_in_driveway": True, "plate_visible": True, "what": "a white SUV", "at": "12:29 am"})
        services.note(services.LIGHTNING, 0.42)
        state = scene.dashboard.state()
        self.assertEqual(state["lot_name"], "Elm Street lot")
        self.assertEqual(state["price"], {"level": "busy", "words": "$8.00 an hour", "until": state["price"]["until"]})
        self.assertEqual(state["light"], "amber")
        self.assertEqual(state["camera"]["state"], "live")
        self.assertEqual(state["last_read"]["plate"], PLATE)
        self.assertEqual(state["scene"]["report"]["what"], "a white SUV")
        stay = state["stay"]
        self.assertEqual((stay["plate"], stay["driver"], stay["so_far"], stay["overstaying"], stay["fee"]), (PLATE, "Dana Driver", "25 minutes", True, "$5.00"))
        kinds = [row["kind"] for row in state["timeline"]]
        self.assertEqual(kinds, ["PRICE_CHECK", "CAMERA_OFFLINE", "ARRIVED"])  # newest first
        arrived = state["timeline"][2]
        self.assertEqual(arrived["how"], "clean")
        self.assertEqual([c["tool"] for c in arrived["calls"]], ["lookup_booking", "set_light", "send_alert"])
        self.assertEqual(arrived["calls"][1]["result"]["light"], "green")
        self.assertEqual(arrived["closing"], "Started the stay, lit green, alerted the driver.")
        self.assertEqual(state["timeline"][0]["closing"], "Tonight: busy")
        self.assertEqual(state["timeline"][1]["how"], "by code")
        names = [s["name"] for s in state["services"]]
        self.assertEqual(names, list(services.SERVICES))
        lightning = state["services"][0]
        self.assertEqual((lightning["recent"], lightning["last_latency_s"]), (True, 0.42))
        self.assertFalse(state["services"][1]["recent"])
        self.assertEqual(state["today"], {"stays": 1, "revenue": "$0.00", "fees_waived": "$0.00", "fees_waived_count": 0, "disputes": 0})
        self.assertIsNone(state["unreadable"])

    def test_spot_empty_offline_camera_and_unreadable_vehicle(self):
        scene = self.scene
        db = Database(scene.db_path)
        db.close_session(scene.session, NOW - timedelta(minutes=1), "data/snapshots/last.jpg")
        db.set_amount(scene.session, 600)
        db.close()
        scene.live.set_camera("offline", NOW - timedelta(seconds=90), "no frames")
        state = scene.dashboard.state()
        self.assertIsNone(state["stay"])
        self.assertEqual(state["camera"]["for"], "1 minute 30 seconds")
        self.assertEqual(state["today"]["revenue"], "$6.00")
        scene.live.set_unreadable("a dark van")
        self.assertEqual(scene.dashboard.state()["unreadable"], "a dark van")
        self.assertEqual(scene.dashboard.state()["scene"], {"enabled": False, "report": None})


class ServerTest(unittest.TestCase):
    def setUp(self):
        services.reset()
        self.scene = Scene()
        self.url = self.scene.dashboard.start()

    def tearDown(self):
        self.scene.dashboard.stop()
        self.scene.close()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=5) as response:
            return response.status, response.headers.get("Content-Type"), response.read()

    def test_page_state_and_frame_are_served_and_nothing_else(self):
        status, content_type, body = self.get("/")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn(b"/state.json", body)
        self.assertIn(b"Timeline", body)
        status, content_type, body = self.get("/state.json")
        self.assertEqual((status, content_type), (200, "application/json"))
        self.assertEqual(json.loads(body)["lot_name"], "Elm Street lot")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/frame.jpg")
        self.assertEqual(caught.exception.code, 404)  # no frame yet
        frame = np.full((1440, 2560, 3), 60, dtype=np.uint8)
        self.scene.live.set_frame(frame, NOW)
        status, content_type, body = self.get("/frame.jpg")
        self.assertEqual((status, content_type), (200, "image/jpeg"))
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((decoded.shape[1], decoded.shape[0]), (960, 540))
        self.assertEqual(self.get("/frame.jpg?t=1")[2], body)  # re-encoded at most once a second
        for path in ("/admin", "/state.json/x", "/reset"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
