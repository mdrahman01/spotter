"""Text messages sent to the Telegram bot, read by long polling on a thread of their own.

Only messages from TELEGRAM_CHAT_ID count, and only those sent after the
inbox started: the bot's old updates are skipped on start. Each text is handed
to `handle(text, received_at)` on the inbox thread, so the watcher never waits
for a reply to be worked out. A failed poll is printed once and retried.
"""

import json
import os
import threading
import time
import urllib.parse
from collections.abc import Callable
from datetime import datetime, timezone

from spotter.alerts import TELEGRAM_API, TelegramSetupError, http_json

POLL_TIMEOUT_S = 25  # Telegram holds the request this long when there is nothing new
RETRY_S = 5


class TelegramInbox:
    def __init__(
        self,
        token: str,
        chat_id: str | int,
        handle: Callable[[str, datetime], None],
        say: Callable[[str], None] = print,
        fetch: Callable[[str], dict] = http_json,
        poll_timeout_s: int = POLL_TIMEOUT_S,
        started_at: float | None = None,
    ) -> None:
        self._token = token
        self.chat_id = str(chat_id)
        self.handle = handle
        self.say = say
        self._fetch = fetch
        self.poll_timeout_s = poll_timeout_s
        self.started_at = time.time() if started_at is None else started_at
        self.offset: int | None = None
        self.received = 0
        self.ignored = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._complained = False

    @classmethod
    def from_env(cls, handle: Callable[[str, datetime], None], say: Callable[[str], None] = print) -> "TelegramInbox":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            raise TelegramSetupError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are needed to read messages")
        return cls(token, chat_id, handle, say)

    def _url(self, **params) -> str:
        return f"{TELEGRAM_API}/bot{self._token}/getUpdates?" + urllib.parse.urlencode(params)

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "[REDACTED]")

    def skip_backlog(self) -> None:
        """Start after the newest update the bot already has, so old messages are not answered."""
        reply = self._fetch(self._url(offset=-1, timeout=0))
        updates = reply.get("result") or [] if reply.get("ok") else []
        self.offset = (updates[-1]["update_id"] + 1) if updates else None

    def poll_once(self, timeout_s: int | None = None) -> list[tuple[str, datetime]]:
        """One getUpdates call; returns the (text, received_at) of the messages that count."""
        params = {"timeout": self.poll_timeout_s if timeout_s is None else timeout_s, "allowed_updates": json.dumps(["message"])}
        if self.offset is not None:
            params["offset"] = self.offset
        try:
            reply = self._fetch(self._url(**params))
            if not reply.get("ok"):
                raise RuntimeError(reply.get("description") or "Telegram answered without ok")
        except Exception as err:
            if not self._complained:
                self._complained = True
                self.say(f"inbox: getUpdates failed, retrying every {RETRY_S}s: {self._redact(f'{type(err).__name__}: {err}')}")
            return []
        self._complained = False
        texts = []
        for update in reply.get("result") or []:
            self.offset = update["update_id"] + 1
            message = update.get("message") or {}
            chat = str((message.get("chat") or {}).get("id", ""))
            text = message.get("text")
            sent = message.get("date") or 0
            if chat != self.chat_id or not isinstance(text, str) or sent < self.started_at:
                self.ignored += 1
                continue
            self.received += 1
            texts.append((text, datetime.fromtimestamp(sent, timezone.utc)))
        return texts

    def _run(self) -> None:
        try:
            self.skip_backlog()
        except Exception as err:
            self.say(f"inbox: could not read the bot's backlog: {self._redact(f'{type(err).__name__}: {err}')}")
        while not self._stop.is_set():
            texts = self.poll_once()
            if not texts and self._complained:
                self._stop.wait(RETRY_S)
            for text, received_at in texts:
                if self._stop.is_set():
                    return
                try:
                    self.handle(text, received_at)
                except Exception as err:  # a bad message must not stop the inbox
                    self.say(f"inbox: handling a message failed: {self._redact(f'{type(err).__name__}: {err}')}")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="telegram-inbox", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
