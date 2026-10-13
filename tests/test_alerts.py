"""Tests of the alert backends with no network: the Telegram HTTP calls are faked."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spotter import config
from spotter.alerts import (
    Alert,
    ConsoleAlerts,
    TelegramAlert,
    TelegramSetupError,
    caption_for,
    find_chat_id,
    make_alerts,
    multipart,
    shrink_jpeg,
)

TOKEN = "123456:FAKE-TOKEN-FOR-TESTS"


def quiet(line: str) -> None:
    pass


class FakeTelegram:
    """Records what would have been posted and answers like the Bot API."""

    def __init__(self, ok=True, raise_error=None, photos_ok=True):
        self.ok = ok
        self.raise_error = raise_error
        self.photos_ok = photos_ok
        self.posts: list[tuple[str, bytes, str]] = []

    def __call__(self, url, body, content_type):
        self.posts.append((url, body, content_type))
        if self.raise_error:
            raise self.raise_error
        if not self.ok:
            return {"ok": False, "description": "Bad Request: chat not found"}
        if url.endswith("/sendMediaGroup"):
            if not self.photos_ok:
                return {"ok": False, "description": "Bad Request: media too large"}
            return {"ok": True, "result": [{"message_id": 43}, {"message_id": 44}]}
        if url.endswith("/sendPhoto") and not self.photos_ok:
            return {"ok": False, "description": "Bad Request: photo too large"}
        return {"ok": True, "result": {"message_id": 42 if url.endswith("/sendMessage") else 43}}

    def methods(self) -> list[str]:
        return [url.rsplit("/", 1)[1] for url, _, _ in self.posts]


def write_image(directory: Path, name: str, width: int, height: int) -> str:
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    cv2.imwrite(str(directory / name), image)
    return name


class CaptionAndPhotoTest(unittest.TestCase):
    def test_caption_has_the_role_prefix(self):
        self.assertEqual(caption_for(Alert("owner", None, "An unbooked car is in the spot.", "x.jpg")),
                         "Owner: An unbooked car is in the spot.")
        self.assertEqual(caption_for(Alert("driver", "Dana", "Welcome.", "")), "Driver: Welcome.")
        self.assertEqual(len(caption_for(Alert("owner", None, "x" * 2000, ""))), 1024)

    def test_snapshot_is_shrunk_to_1280_wide(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            big = write_image(directory, "big.jpg", 1920, 1080)
            small = write_image(directory, "small.jpg", 640, 480)
            for name, expected in ((big, (1280, 720)), (small, (640, 480))):
                decoded = cv2.imdecode(np.frombuffer(shrink_jpeg(directory / name), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual((decoded.shape[1], decoded.shape[0]), expected)
            self.assertIsNone(shrink_jpeg(directory / "missing.jpg"))

    def test_multipart_body(self):
        body, content_type = multipart({"chat_id": "7", "caption": "Owner: hi"}, "photo", "snapshot.jpg", b"\xff\xd8JPEG")
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertEqual(body.count(f"--{boundary}".encode()), 4)  # 3 parts and the closing marker
        self.assertIn(b'name="caption"\r\n\r\nOwner: hi\r\n', body)
        self.assertIn(b'name="photo"; filename="snapshot.jpg"\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8JPEG\r\n', body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))


class TelegramAlertTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        write_image(self.root / "data", "snap.jpg", 1920, 1080)

    def tearDown(self):
        self.tmp.cleanup()

    def alerts(self, fake):
        return TelegramAlert(TOKEN, 7, say=quiet, post=fake, repo_root=self.root)

    def test_the_message_goes_first_as_text_and_the_photo_follows(self):
        fake = FakeTelegram()
        telegram = self.alerts(fake)
        delivery = telegram.send(Alert("owner", None, "An unbooked car is in the spot.", "data/snap.jpg"))
        self.assertTrue(delivery.delivered)
        self.assertEqual((delivery.message_id, delivery.photos), (42, "queued"))
        url, body, content_type = fake.posts[0]  # the text is already there when send() returns
        self.assertTrue(url.endswith("/sendMessage"))
        self.assertIn(TOKEN, url)  # the token belongs in the URL, and nowhere printed
        self.assertEqual(json.loads(body), {"chat_id": "7", "text": "Owner: An unbooked car is in the spot."})
        results = telegram.flush()
        self.assertEqual(fake.methods(), ["sendMessage", "sendPhoto"])
        url, body, content_type = fake.posts[1]
        self.assertTrue(content_type.startswith("multipart/form-data"))
        self.assertIn(b"Owner: photo", body)
        photo = body.split(b"Content-Type: image/jpeg\r\n\r\n")[1].rsplit(b"\r\n--", 1)[0]
        decoded = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[1], 1280)
        self.assertEqual([(r.to, r.paths, r.delivered, r.message_id) for r in results], [("owner", ("data/snap.jpg",), True, 43)])
        self.assertEqual(telegram.flush(), [])  # reported once
        telegram.close()

    def test_several_photos_go_as_one_album(self):
        write_image(self.root / "data", "two.jpg", 640, 480)
        fake = FakeTelegram()
        telegram = self.alerts(fake)
        delivery = telegram.send(Alert("driver", "Dana", "Here is the proof.", "data/snap.jpg", ("data/two.jpg", "data/snap.jpg", "data/gone.jpg")))
        self.assertEqual(delivery.photos, "queued")
        results = telegram.flush()
        self.assertEqual(fake.methods(), ["sendMessage", "sendMediaGroup"])
        body = fake.posts[1][1]
        self.assertEqual(body.count(b"Content-Type: image/jpeg"), 2)  # the missing file is left out, duplicates once
        self.assertIn(b"attach://photo0", body)
        self.assertEqual(results[0].paths, ("data/snap.jpg", "data/two.jpg", "data/gone.jpg"))
        self.assertTrue(results[0].delivered)
        telegram.close()

    def test_text_only_when_there_is_no_snapshot(self):
        fake = FakeTelegram()
        telegram = self.alerts(fake)
        delivery = telegram.send(Alert("driver", "Dana", "Welcome.", ""))
        self.assertEqual((delivery.delivered, delivery.photos), (True, "none"))
        self.assertEqual(telegram.flush(), [])
        self.assertEqual(fake.methods(), ["sendMessage"])

    def test_a_missing_snapshot_is_a_photo_failure_not_a_message_failure(self):
        fake = FakeTelegram()
        telegram = self.alerts(fake)
        delivery = telegram.send(Alert("driver", "Dana", "Welcome.", "data/gone.jpg"))
        self.assertTrue(delivery.delivered)
        results = telegram.flush()
        self.assertEqual(fake.methods(), ["sendMessage"])
        self.assertFalse(results[0].delivered)
        self.assertIn("no readable photo", results[0].error)

    def test_failures_are_reported_without_raising_or_leaking_the_token(self):
        rejected = self.alerts(FakeTelegram(ok=False))
        delivery = rejected.send(Alert("owner", None, "hi", "data/snap.jpg"))
        self.assertFalse(delivery.delivered)
        self.assertIn("chat not found", delivery.error)
        self.assertEqual(rejected.flush(), [])  # no photo is queued behind a message that failed
        boom = FakeTelegram(raise_error=OSError(f"connection refused for https://api.telegram.org/bot{TOKEN}/sendMessage"))
        printed = []
        broken = TelegramAlert(TOKEN, 7, say=printed.append, post=boom, repo_root=self.root)
        delivery = broken.send(Alert("owner", None, "hi", "data/snap.jpg"))
        self.assertTrue(delivery.error.startswith("OSError:"))
        self.assertNotIn(TOKEN, delivery.error)
        self.assertNotIn(TOKEN, "".join(printed))
        self.assertEqual(len(broken.failed), 1)
        self.assertEqual(broken.sent, [])
        photo_fails = FakeTelegram(photos_ok=False)
        printed = []
        telegram = TelegramAlert(TOKEN, 7, say=printed.append, post=photo_fails, repo_root=self.root)
        self.assertTrue(telegram.send(Alert("owner", None, "hi", "data/snap.jpg")).delivered)
        results = telegram.flush()
        self.assertFalse(results[0].delivered)
        self.assertIn("photo too large", results[0].error)
        self.assertNotIn(TOKEN, "".join(printed))

    def test_chat_id_comes_from_the_latest_private_chat(self):
        updates = {"ok": True, "result": [
            {"update_id": 1, "message": {"chat": {"id": 111, "type": "group"}, "text": "hi"}},
            {"update_id": 2, "message": {"chat": {"id": 555, "type": "private"}, "text": "/start"}},
            {"update_id": 3, "my_chat_member": {"chat": {"id": 666, "type": "private"}}},
        ]}
        self.assertEqual(find_chat_id(TOKEN, fetch=lambda url: updates), 666)
        self.assertIsNone(find_chat_id(TOKEN, fetch=lambda url: {"ok": True, "result": []}))
        with self.assertRaises(TelegramSetupError) as caught:
            find_chat_id(TOKEN, fetch=lambda url: {"ok": False, "description": "Unauthorized"})
        self.assertIn("Unauthorized", str(caught.exception))


class ConsoleAlertsTest(unittest.TestCase):
    def test_console_prints_the_photos_with_the_message(self):
        printed = []
        console = ConsoleAlerts(say=printed.append)
        delivery = console.send(Alert("owner", None, "Look.", "data/a.jpg", ("data/b.jpg",)))
        self.assertEqual((delivery.delivered, delivery.photos), (True, "attached"))
        self.assertEqual(printed, ["ALERT to owner: Look. [photos: data/a.jpg, data/b.jpg]"])
        self.assertEqual(console.send(Alert("driver", "Dana", "Hi.", "")).photos, "none")
        self.assertEqual(console.flush(), [])

    def test_attachments_are_the_snapshot_first_then_the_photos_at_most_three(self):
        alert = Alert("owner", None, "x", "s.jpg", ("a.jpg", "s.jpg", "b.jpg", "c.jpg"))
        self.assertEqual(alert.attachments, ("s.jpg", "a.jpg", "b.jpg"))
        self.assertEqual(Alert("owner", None, "x", "").attachments, ())


class BackendChoiceTest(unittest.TestCase):
    def test_console_is_the_default_and_telegram_is_a_setting(self):
        self.assertEqual(config.load({}, env={}).alerts, "console")
        self.assertEqual(config.load({}, env={"SPOTTER_ALERTS": "Telegram"}).alerts, "telegram")
        with self.assertRaises(ValueError):
            config.load({}, env={"SPOTTER_ALERTS": "pigeon"})
        self.assertIsInstance(make_alerts("console", quiet), ConsoleAlerts)
        with self.assertRaises(ValueError):
            make_alerts("pigeon", quiet)

    def test_minimum_charge_setting(self):
        self.assertEqual(config.load({}, env={}).min_charge_cents, 100)
        self.assertEqual(config.load({}, env={"SPOTTER_MIN_CHARGE_CENTS": "250"}).min_charge_cents, 250)
        with self.assertRaises(ValueError):
            config.load({}, env={"SPOTTER_MIN_CHARGE_CENTS": "-1"})


if __name__ == "__main__":
    unittest.main()
