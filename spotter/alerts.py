"""Alerts to the owner or a driver: printed to the console, or sent to your Telegram chat.

SPOTTER_ALERTS=console (the default; tests need no network) prints lines like
"ALERT to owner: ...". SPOTTER_ALERTS=telegram sends each alert to your own
chat with the bot in TELEGRAM_BOT_TOKEN. The message goes first, as text, and
the call returns as soon as Telegram has it (well under a second on a home
link); the photos follow from a background thread, so neither the model nor
the light ever waits for an upload. The text starts with "Owner:" or "Driver:"
so one chat can play both roles. The snapshots show a real plate and the
street, so they go to that one chat and nowhere else.

The chat id is found once with getUpdates and saved to .env as
TELEGRAM_CHAT_ID. A failed send never raises: it is printed and reported in
the tool result, and the attendant carries on. The result of each photo
upload is kept until flush() is called, so the finish check can record it.
"""

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
from dotenv import set_key

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
TELEGRAM_API = "https://api.telegram.org"
PHOTO_WIDTH = 1280
CAPTION_LIMIT = 1024  # Telegram's limit for a photo caption
TEXT_LIMIT = 4096  # and for a text message
MAX_PHOTOS = 3  # per alert: the event's frame, or a dispute's proof
PHOTO_WAIT_S = 30.0  # how long flush() waits for queued photos


@dataclass(frozen=True)
class Alert:
    to: str  # "owner" or "driver"
    name: str | None  # the driver's name, if known
    message: str
    snapshot: str  # repo-relative path of the event's frame, attached by the code ("" for none)
    photos: tuple[str, ...] = ()  # further photos, by repo-relative path (a dispute's proof)

    @property
    def attachments(self) -> tuple[str, ...]:
        """Every photo to send after the message, the event's frame first, at most MAX_PHOTOS."""
        seen: list[str] = []
        for path in (self.snapshot, *self.photos):
            if path and path not in seen:
                seen.append(path)
        return tuple(seen[:MAX_PHOTOS])


@dataclass(frozen=True)
class Delivery:
    """What happened to one alert's message; its photos are reported later by flush()."""

    error: str | None  # None when the message reached the chat
    message_id: int | None = None
    photos: str = "none"  # "none", "attached" (shown with the message) or "queued" (following it)

    @property
    def delivered(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class PhotoResult:
    """The outcome of the photos queued for one alert."""

    to: str
    paths: tuple[str, ...]
    error: str | None
    message_id: int | None = None

    @property
    def delivered(self) -> bool:
        return self.error is None


class Alerts(Protocol):
    def send(self, alert: Alert) -> Delivery:
        """Deliver one alert's message now; any photos may follow."""

    def flush(self, timeout_s: float = PHOTO_WAIT_S) -> list[PhotoResult]:
        """Wait for queued photos (at most timeout_s) and return their results since the last flush."""

    def close(self) -> None:
        """Send what is still queued and stop."""


class ConsoleAlerts:
    """Prints "ALERT to owner: ..." instead of messaging anyone, and keeps what was sent."""

    def __init__(self, say: Callable[[str], None] = print) -> None:
        self.say = say
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> Delivery:
        self.sent.append(alert)
        who = alert.to if alert.name is None else f"{alert.to} ({alert.name})"
        photos = ", ".join(alert.attachments) or "none"
        self.say(f"ALERT to {who}: {alert.message} [photos: {photos}]")
        return Delivery(None, None, "attached" if alert.attachments else "none")

    def flush(self, timeout_s: float = PHOTO_WAIT_S) -> list[PhotoResult]:
        return []

    def close(self) -> None:
        pass


class TelegramSetupError(Exception):
    """The Telegram alerts cannot start: a missing token, or no chat to send to."""


def caption_for(alert: Alert) -> str:
    """The message with its role prefix, within Telegram's caption limit."""
    return f"{alert.to.capitalize()}: {alert.message}"[:CAPTION_LIMIT]


def shrink_jpeg(path: Path, width: int = PHOTO_WIDTH) -> bytes | None:
    """The image at `path` as JPEG bytes, at most `width` pixels wide; None if unreadable."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, current = image.shape[:2]
    if current > width:
        image = cv2.resize(image, (width, round(height * width / current)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes() if ok else None


def multipart_files(fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """A multipart/form-data body with text fields and JPEG files (field, filename, bytes)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for field, filename, content in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n".encode() + content + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """A multipart/form-data body with text fields and one JPEG file; returns (body, content type)."""
    return multipart_files(fields, [(file_field, filename, content)])


def http_json(url: str, body: bytes | None = None, content_type: str | None = None) -> dict:
    """POST (or GET when body is None) and return Telegram's JSON reply, also for HTTP errors."""
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read())
        except ValueError:
            return {"ok": False, "description": f"HTTP {err.code}"}


def find_chat_id(token: str, fetch: Callable[[str], dict] = http_json) -> int | None:
    """The chat the bot was last written to, from getUpdates; None if nobody has yet.

    Raises TelegramSetupError if the Bot API rejects the call (bad token, or a
    webhook is set, which blocks getUpdates).
    """
    reply = fetch(f"{TELEGRAM_API}/bot{token}/getUpdates")
    if not reply.get("ok"):
        raise TelegramSetupError(f"getUpdates failed: {reply.get('description') or 'no description'}")
    chats = []
    for update in reply.get("result") or []:
        for key in ("message", "edited_message", "channel_post", "my_chat_member"):
            chat = (update.get(key) or {}).get("chat")
            if chat and "id" in chat:
                chats.append(chat)
    private = [chat for chat in chats if chat.get("type") == "private"] or chats
    return private[-1]["id"] if private else None


class TelegramAlert:
    """Sends each alert's message to one Telegram chat at once, and its photos right after."""

    def __init__(
        self,
        token: str,
        chat_id: str | int,
        say: Callable[[str], None] = print,
        post: Callable[[str, bytes, str], dict] = http_json,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self._token = token
        self.chat_id = str(chat_id)
        self.say = say
        self._post = post
        self.repo_root = repo_root
        self.sent: list[Alert] = []
        self.failed: list[tuple[Alert, str]] = []
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._results: list[PhotoResult] = []
        self._worker: threading.Thread | None = None

    @classmethod
    def from_env(cls, say: Callable[[str], None] = print) -> "TelegramAlert":
        """Build from TELEGRAM_BOT_TOKEN, finding and saving TELEGRAM_CHAT_ID on first use."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise TelegramSetupError("TELEGRAM_BOT_TOKEN is missing from .env")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not chat_id:
            found = find_chat_id(token)
            if found is None:
                raise TelegramSetupError(
                    "getUpdates shows no chat yet: open the bot's chat in Telegram, press Start, then run again"
                )
            chat_id = str(found)
            set_key(ENV_PATH, "TELEGRAM_CHAT_ID", chat_id, quote_mode="never")
            os.environ["TELEGRAM_CHAT_ID"] = chat_id
            say("Telegram chat id found with getUpdates and saved to .env as TELEGRAM_CHAT_ID")
        return cls(token, chat_id, say)

    def _url(self, method: str) -> str:
        return f"{TELEGRAM_API}/bot{self._token}/{method}"

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "[REDACTED]")

    def _call(self, method: str, body: bytes, content_type: str) -> dict:
        """One Bot API call; raises RuntimeError when Telegram says no."""
        reply = self._post(self._url(method), body, content_type)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("description") or "Telegram answered without ok")
        return reply.get("result") or {}

    # ------------------------------------------------------------------ message

    def send(self, alert: Alert) -> Delivery:
        text = caption_for(alert)[:TEXT_LIMIT]
        try:
            result = self._call("sendMessage", json.dumps({"chat_id": self.chat_id, "text": text}).encode(), "application/json")
        except Exception as err:  # network, HTTP, or an unexpected reply: never take the attendant down
            error = self._redact(f"{type(err).__name__}: {err}")
            self.failed.append((alert, error))
            self.say(f"ALERT to {alert.to} NOT delivered by Telegram: {error}")
            return Delivery(error)
        self.sent.append(alert)
        message_id = result.get("message_id")
        photos = alert.attachments
        self.say(
            f"ALERT to {alert.to} by Telegram (text, message_id {message_id}): {text}"
            + (f" [photos to follow: {', '.join(photos)}]" if photos else "")
        )
        if photos:
            self._start_worker()
            self._queue.put((alert, photos))
        return Delivery(None, message_id, "queued" if photos else "none")

    # ------------------------------------------------------------------- photos

    def _start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._send_photos_forever, name="telegram-photos", daemon=True)
            self._worker.start()

    def _send_photos_forever(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                alert, paths = item
                result = self._send_photos(alert, paths)
                with self._lock:
                    self._results.append(result)
            finally:
                self._queue.task_done()

    def _send_photos(self, alert: Alert, paths: tuple[str, ...]) -> PhotoResult:
        """Upload the alert's photos in one message: sendPhoto for one, sendMediaGroup for more."""
        jpegs = [(path, shrink_jpeg(self.repo_root / path)) for path in paths]
        missing = [path for path, jpeg in jpegs if jpeg is None]
        jpegs = [(path, jpeg) for path, jpeg in jpegs if jpeg is not None]
        role = alert.to.capitalize()
        try:
            if not jpegs:
                raise FileNotFoundError(f"no readable photo among {', '.join(paths)}")
            if len(jpegs) == 1:
                body, content_type = multipart({"chat_id": self.chat_id, "caption": f"{role}: photo"}, "photo", "photo.jpg", jpegs[0][1])
                result = self._call("sendPhoto", body, content_type)
                message_id = result.get("message_id")
            else:
                media = [
                    {"type": "photo", "media": f"attach://photo{i}", **({"caption": f"{role}: photos"} if i == 0 else {})}
                    for i in range(len(jpegs))
                ]
                files = [(f"photo{i}", f"photo{i}.jpg", jpeg) for i, (_, jpeg) in enumerate(jpegs)]
                body, content_type = multipart_files({"chat_id": self.chat_id, "media": json.dumps(media)}, files)
                result = self._call("sendMediaGroup", body, content_type)
                message_id = (result[0].get("message_id") if isinstance(result, list) and result else None)
        except Exception as err:
            error = self._redact(f"{type(err).__name__}: {err}")
            self.say(f"PHOTOS for {alert.to} NOT delivered by Telegram: {error}")
            return PhotoResult(alert.to, paths, error)
        note = f" (unreadable, left out: {', '.join(missing)})" if missing else ""
        self.say(f"PHOTOS for {alert.to} by Telegram (message_id {message_id}): {', '.join(p for p, _ in jpegs)}{note}")
        return PhotoResult(alert.to, paths, None, message_id)

    def flush(self, timeout_s: float = PHOTO_WAIT_S) -> list[PhotoResult]:
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        with self._lock:
            results, self._results = self._results, []
        return results

    def close(self) -> None:
        self.flush()
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=5)


def make_alerts(kind: str, say: Callable[[str], None] = print) -> Alerts:
    """The alerts backend named by the SPOTTER_ALERTS setting."""
    if kind == "console":
        return ConsoleAlerts(say)
    if kind == "telegram":
        return TelegramAlert.from_env(say)
    raise ValueError(f"unknown alerts backend {kind!r}")
