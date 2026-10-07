"""Alerts to the owner or a driver: printed to the console, or sent to your Telegram chat.

SPOTTER_ALERTS=console (the default; tests need no network) prints lines like
"ALERT to owner: ...". SPOTTER_ALERTS=telegram sends each alert to your own
chat with the bot in TELEGRAM_BOT_TOKEN: as a photo with a caption when the
event has a snapshot, otherwise as text. The caption starts with "Owner:" or
"Driver:" so one chat can play both roles. The snapshots show a real plate and
the street, so they go to that one chat and nowhere else.

The chat id is found once with getUpdates and saved to .env as
TELEGRAM_CHAT_ID. A failed send never raises: it is printed and reported in the
tool result, and the attendant carries on.
"""

import json
import os
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


@dataclass(frozen=True)
class Alert:
    to: str  # "owner" or "driver"
    name: str | None  # the driver's name, if known
    message: str
    snapshot: str  # repo-relative path of the event's snapshot, attached by the code


class Alerts(Protocol):
    def send(self, alert: Alert) -> str | None:
        """Deliver one alert. Returns None, or a description of why it was not delivered."""


class ConsoleAlerts:
    """Prints "ALERT to owner: ..." instead of messaging anyone, and keeps what was sent."""

    def __init__(self, say: Callable[[str], None] = print) -> None:
        self.say = say
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> str | None:
        self.sent.append(alert)
        who = alert.to if alert.name is None else f"{alert.to} ({alert.name})"
        self.say(f"ALERT to {who}: {alert.message} [snapshot: {alert.snapshot}]")
        return None


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


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """A multipart/form-data body with text fields and one JPEG file; returns (body, content type)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode() + content + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


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
    """Sends each alert to one Telegram chat, as a photo with a caption when there is a snapshot."""

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

    def send(self, alert: Alert) -> str | None:
        caption = caption_for(alert)
        photo = shrink_jpeg(self.repo_root / alert.snapshot) if alert.snapshot else None
        try:
            if photo:
                body, content_type = multipart(
                    {"chat_id": self.chat_id, "caption": caption}, "photo", "snapshot.jpg", photo
                )
                reply = self._post(self._url("sendPhoto"), body, content_type)
            else:
                body = json.dumps({"chat_id": self.chat_id, "text": caption}).encode()
                reply = self._post(self._url("sendMessage"), body, "application/json")
            if not reply.get("ok"):
                raise RuntimeError(reply.get("description") or "Telegram answered without ok")
        except Exception as err:  # network, HTTP, or an unexpected reply: never take the attendant down
            error = self._redact(f"{type(err).__name__}: {err}")
            self.failed.append((alert, error))
            self.say(f"ALERT to {alert.to} NOT delivered by Telegram: {error}")
            return error
        self.sent.append(alert)
        kind = "photo" if photo else "text"
        message_id = (reply.get("result") or {}).get("message_id")
        self.say(
            f"ALERT to {alert.to} by Telegram ({kind}, message_id {message_id}): {caption}"
            f" [snapshot: {alert.snapshot or 'none'}]"
        )
        return None


def make_alerts(kind: str, say: Callable[[str], None] = print) -> Alerts:
    """The alerts backend named by the SPOTTER_ALERTS setting."""
    if kind == "console":
        return ConsoleAlerts(say)
    if kind == "telegram":
        return TelegramAlert.from_env(say)
    raise ValueError(f"unknown alerts backend {kind!r}")
