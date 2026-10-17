"""The owner's page: one HTML file served from the worker, fed by a JSON endpoint and a frame.

    python -m spotter.run --dashboard            (or SPOTTER_DASHBOARD=on)

GET /            the page (spotter/dashboard.html, styles and script inside)
GET /state.json  everything the page shows, polled every 2 s
GET /frame.jpg   the newest camera frame, 960 px wide with the watch zone drawn,
                 re-encoded at most once a second; 404 until the first frame. The page
                 shows "Waiting for the camera" until then, and when the camera drops it
                 keeps the last frame under an "offline" label.

The page can change nothing: there are no other routes. It works during a
replay too. The watcher updates a LiveState (frame, last read, camera state,
scene report); the rest comes from the database and the services registry
on the server's own thread.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from spotter import services
from spotter.config import Settings
from spotter.db import Database, parse
from spotter.reader import zone_box
from spotter.tools import dollars
from spotter.words import clock_words, duration_words

PAGE = Path(__file__).resolve().parent / "dashboard.html"
FRAME_WIDTH = 960
FRAME_EVERY_S = 1.0
TIMELINE_ROWS = 60
QUIET_TOOLS = ("words", "finish", "reminder", "final", "model", "dispute", "camera", "photo", "run")
CODE_ROWS = ("CAMERA_OFFLINE", "CAMERA_BACK", "RUN_STARTED", "RUN_STOPPED")  # code's own rows: nothing to finish


@dataclass
class LiveState:
    """What only the watcher knows, updated from its thread and read by the server's."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: np.ndarray | None = None
    frame_at: datetime | None = None
    last_read: tuple[str, datetime] | None = None
    camera: str = "starting"  # starting, live, offline, replay
    camera_since: datetime | None = None
    camera_detail: str = ""
    scene_enabled: bool = False
    scene_report: dict | None = None  # vehicle_in_driveway, plate_visible, what, at
    unreadable_what: str | None = None
    replay_name: str | None = None

    def set_frame(self, frame: np.ndarray, at: datetime) -> None:
        with self.lock:
            self.frame, self.frame_at = frame, at

    def set_read(self, plate: str, at: datetime) -> None:
        with self.lock:
            self.last_read = (plate, at)

    def set_camera(self, state: str, since: datetime | None = None, detail: str = "") -> None:
        with self.lock:
            self.camera, self.camera_since, self.camera_detail = state, since, detail

    def set_scene(self, enabled: bool, report: dict | None = None) -> None:
        with self.lock:
            self.scene_enabled = enabled
            if report is not None:
                self.scene_report = report

    def set_unreadable(self, what: str | None) -> None:
        with self.lock:
            self.unreadable_what = what


class Dashboard:
    def __init__(self, settings: Settings, live: LiveState, light, db_path=None, now=None) -> None:
        self.settings = settings
        self.live = live
        self.light = light
        self.db_path = db_path
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._jpeg: bytes | None = None
        self._jpeg_at = 0.0
        self._jpeg_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = settings.dashboard_port

    # ------------------------------------------------------------------ state

    def _db(self) -> Database:
        return Database(self.db_path) if self.db_path else Database()

    def state(self) -> dict:
        now = self._now()
        db = self._db()
        try:
            return self._state(db, now)
        finally:
            db.close()

    def _state(self, db: Database, now: datetime) -> dict:
        settings = self.settings
        price = db.tonight_price(now)
        if price:
            price_part = {"level": price["level"], "words": f"{dollars(price['rate_cents_per_hour'])} an hour", "until": clock_words(parse(price["valid_until"]), now)}
        else:
            price_part = {"level": "base", "words": f"{dollars(settings.rate_base)} an hour", "until": None}

        with self.live.lock:
            camera, since, detail = self.live.camera, self.live.camera_since, self.live.camera_detail
            last_read = self.live.last_read
            scene_enabled, scene_report = self.live.scene_enabled, self.live.scene_report
            unreadable = self.live.unreadable_what
            frame_at = self.live.frame_at
            replay_name = self.live.replay_name
        camera_part = {"state": camera, "detail": detail, "since": clock_words(since, now) if since else None,
                       "for": duration_words((now - since).total_seconds()) if since and camera == "offline" else None,
                       "replay": replay_name}

        stay = db.open_sessions()
        stay_part = None
        if stay:
            session = stay[0]
            booking = db.booking(session["booking_id"])
            arrived = parse(session["arrived_at"])
            stay_part = {
                "plate": session["plate"],
                "driver": booking["driver"] if booking else None,
                "arrived": clock_words(arrived, now),
                "so_far": duration_words((now - arrived).total_seconds()),
                "booking_ends": clock_words(parse(booking["ends_at"]), now) if booking else None,
                "overstaying": bool(session["overstayed_at"]),
                "fee": dollars(settings.overstay_fee_cents) if session["overstayed_at"] else None,
                "ending_soon": bool(session["ending_soon_at"]),
            }

        last = db.latest_closed_session() if not stay else None
        last_stay = None
        if last:
            last_stay = {"plate": last["plate"], "left": clock_words(parse(last["left_at"]), now),
                         "billed": dollars(last["amount_cents"]) if last["amount_cents"] is not None else None}
        timeline = self._timeline(db, now)
        today_part = self._today(db, now)
        return {
            "generated_at": clock_words(now, now),
            "lot_name": settings.lot_name,
            "price": price_part,
            "light": getattr(self.light, "color", None) or "off",
            "camera": camera_part,
            "frame_at": clock_words(frame_at, now) if frame_at else None,
            "zone": list(settings.zone),
            "last_read": {"plate": last_read[0], "at": clock_words(last_read[1], now)} if last_read else None,
            "scene": {"enabled": scene_enabled, "report": scene_report},
            "stay": stay_part,
            "last_stay": last_stay,
            "unreadable": unreadable,
            "timeline": timeline,
            "services": services.snapshot(),
            "today": today_part,
        }

    def _timeline(self, db: Database, now: datetime) -> list[dict]:
        groups: list[dict] = []
        for action in db.actions():
            event = json.loads(action["event"])
            if not groups or groups[-1]["event"] != event:
                groups.append({"event": event, "rows": []})
            groups[-1]["rows"].append(action)
        rows = []
        for group in groups[-TIMELINE_ROWS:]:
            event, actions = group["event"], group["rows"]
            kind = event.get("type", "?")
            when = parse(event["time"]) if event.get("time") else now
            how, closing, calls, detail = None, None, [], ""
            for action in actions:
                tool = action["tool"]
                result = _load(action["result"])
                arguments = _load(action["arguments"])
                if tool == "finish":
                    how = arguments.get("how")
                elif tool == "final":
                    closing = result.get("text")
                elif tool == "run":  # RUN_STARTED / RUN_STOPPED: the words code wrote for the row
                    detail = arguments.get("detail") or ""
                elif tool == "camera":
                    down = arguments.get("downtime_s")
                    detail = f"down {duration_words(down)}" if down else f"no frames since {clock_words(when, now)}"
                elif tool in QUIET_TOOLS:
                    if tool == "dispute":
                        closing = closing or f"Message: {arguments.get('text', '')!r}"
                    continue
                else:
                    calls.append({
                        "tool": tool,
                        "by_code": arguments.get("by") == "code" or result.get("by") == "code",
                        "arguments": _brief(arguments),
                        "result": _brief(result),
                        "error": result.get("error"),
                    })
            if kind in CODE_ROWS:
                how = "—"  # the page shows a dash: nothing for the attendant to finish, never "by code"
            if kind == "PRICE_CHECK":
                finish = next((_load(a["result"]) for a in actions if a["tool"] == "finish"), {})
                level = finish.get("level")
                how = how or "by code"
                closing = closing or (f"Tonight: {level}" if level else None)
            rows.append({
                "time": clock_words(when, now),
                "kind": kind,
                "plate": event.get("plate") or "",
                "how": how or "",
                "calls": calls,
                "closing": closing or "",
                "detail": detail or event.get("what") or (event.get("stayed_for") or "" if kind == "SPOT_CLEAR" else ""),
            })
        rows.reverse()
        return rows

    def _today(self, db: Database, now: datetime) -> dict:
        start_local = now.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        start = start_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        sessions = [s for s in db.sessions() if s["arrived_at"] >= start]
        revenue = sum(s["amount_cents"] or 0 for s in sessions if s["status"] == "closed")
        waived = sum(s["waived_cents"] or 0 for s in sessions if s["fee_waived_at"])
        disputes = {a["event"] for a in db.actions() if a["time"] >= start and json.loads(a["event"]).get("type") == "DISPUTE"}
        return {"stays": len(sessions), "revenue": dollars(revenue), "fees_waived": dollars(waived), "fees_waived_count": sum(1 for s in sessions if s["fee_waived_at"]),
                "disputes": len(disputes)}

    # ------------------------------------------------------------------ frame

    def frame_jpeg(self) -> bytes | None:
        """The newest frame as JPEG, 960 px wide with the zone drawn, re-encoded at most once a second."""
        with self._jpeg_lock:
            if self._jpeg is not None and time.monotonic() - self._jpeg_at < FRAME_EVERY_S:
                return self._jpeg
            with self.live.lock:
                frame = self.live.frame
            if frame is None:
                return self._jpeg
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = zone_box(self.settings.zone, width, height)
            small = cv2.resize(frame, (FRAME_WIDTH, round(height * FRAME_WIDTH / width)), interpolation=cv2.INTER_AREA)
            scale = FRAME_WIDTH / width
            cv2.rectangle(small, (round(x0 * scale), round(y0 * scale)), (round(x1 * scale), round(y1 * scale)), (80, 220, 120), 2)
            ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._jpeg, self._jpeg_at = encoded.tobytes(), time.monotonic()
            return self._jpeg

    # ----------------------------------------------------------------- server

    def start(self) -> str:
        dashboard = self
        page = PAGE.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # quiet: the console is the attendant's
                pass

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/":
                        self._send(200, "text/html; charset=utf-8", page)
                    elif path == "/state.json":
                        self._send(200, "application/json", json.dumps(dashboard.state()).encode())
                    elif path == "/frame.jpg":
                        jpeg = dashboard.frame_jpeg()
                        if jpeg is None:
                            self._send(404, "text/plain", b"no frame yet")
                        else:
                            self._send(200, "image/jpeg", jpeg)
                    else:
                        self._send(404, "text/plain", b"not found")
                except Exception as err:  # the page must never take the watcher down
                    self._send(500, "text/plain", f"{type(err).__name__}: {err}".encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="dashboard", daemon=True)
        self._thread.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _load(text: str):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


SKIP_KEYS = ("by", "snapshot", "delivery_error", "name", "amount_cents", "last_read_snapshot")


def _plain(item) -> str:
    """A value as the owner would read it: nested objects flattened, booleans as yes/no."""
    if isinstance(item, bool):
        return "yes" if item else "no"
    if isinstance(item, dict):
        return ", ".join(f"{k} {_plain(v)}" for k, v in item.items() if v is not None and k not in SKIP_KEYS)
    if isinstance(item, list):
        return "; ".join(_plain(v) for v in item)
    text = str(item)
    return text if len(text) <= 300 else text[:300] + "…"


def _brief(value: dict) -> dict:
    """Arguments or a result for the page: short plain strings, no empty fields, no file paths."""
    out = {}
    for key, item in value.items():
        if key in SKIP_KEYS or item is None or item == "":
            continue
        if key == "results" and isinstance(item, list):
            item = [f"{r.get('title', '')[:70]} <{r.get('url', '')}>" for r in item]
        if key == "photos" and isinstance(item, list):
            item = [p.get("name", p) if isinstance(p, dict) else p for p in item]
        out[key] = _plain(item)
    return out
