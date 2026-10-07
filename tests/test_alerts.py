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

    def __init__(self, ok=True, raise_error=None):
        self.ok = ok
        self.raise_error = raise_error
        self.posts: list[tuple[str, bytes, str]] = []

    def __call__(self, url, body, content_type):
        self.posts.append((url, body, content_type))
        if self.raise_error:
            raise self.raise_error
        if not self.ok:
            return {"ok": False, "description": "Bad Request: chat not found"}
        return {"ok": True, "result": {"message_id": 42}}


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

    def test_photo_with_caption_when_there_is_a_snapshot(self):
        fake = FakeTelegram()
        error = self.alerts(fake).send(Alert("owner", None, "An unbooked car is in the spot.", "data/snap.jpg"))
        self.assertIsNone(error)
        url, body, content_type = fake.posts[0]
        self.assertTrue(url.endswith("/sendPhoto"))
        self.assertIn(TOKEN, url)  # the token belongs in the URL, and nowhere printed
        self.assertTrue(content_type.startswith("multipart/form-data"))
        self.assertIn(b"Owner: An unbooked car is in the spot.", body)
        photo = body.split(b"Content-Type: image/jpeg\r\n\r\n")[1].rsplit(b"\r\n--", 1)[0]
        decoded = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[1], 1280)

    def test_text_when_there_is_no_snapshot_or_it_is_missing(self):
        fake = FakeTelegram()
        telegram = self.alerts(fake)
        telegram.send(Alert("driver", "Dana", "Welcome.", ""))
        telegram.send(Alert("driver", "Dana", "Welcome.", "data/gone.jpg"))
        for url, body, content_type in fake.posts:
            self.assertTrue(url.endswith("/sendMessage"))
            self.assertEqual(content_type, "application/json")
            self.assertEqual(json.loads(body), {"chat_id": "7", "text": "Driver: Welcome."})

    def test_failures_are_reported_without_raising_or_leaking_the_token(self):
        rejected = self.alerts(FakeTelegram(ok=False))
        error = rejected.send(Alert("owner", None, "hi", "data/snap.jpg"))
        self.assertIn("chat not found", error)
        boom = FakeTelegram(raise_error=OSError(f"connection refused for https://api.telegram.org/bot{TOKEN}/sendPhoto"))
        printed = []
        broken = TelegramAlert(TOKEN, 7, say=printed.append, post=boom, repo_root=self.root)
        error = broken.send(Alert("owner", None, "hi", "data/snap.jpg"))
        self.assertTrue(error.startswith("OSError:"))
        self.assertNotIn(TOKEN, error)
        self.assertNotIn(TOKEN, "".join(printed))
        self.assertEqual(len(broken.failed), 1)
        self.assertEqual(broken.sent, [])

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
