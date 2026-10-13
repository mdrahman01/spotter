"""Cosmos Reason as Spotter's second pair of eyes: one camera frame in, a small report out.

The plate reader only sees plates. A look sends one frame, shrunk to 1280 px
wide, to the Cosmos endpoint and gets back a Report: vehicle_in_driveway,
plate_visible and `what` (twelve words at most). The question names the spot
(SPOTTER_SPOT_DESCRIPTION) and says to ignore vehicles on the street behind,
and the server's structured output forces valid JSON. A failed, slow or
malformed answer is "scene unavailable" (None), never an answer.

SPOTTER_SCENE=off (the default, nothing changes) | cosmos (COSMOS_URL and
COSMOS_TOKEN from .env) | script:<file> (canned reports for tests).

Looks run on a worker thread so the watcher never waits, and nothing acts on
one look: SceneWatch needs two looks in a row that agree before it decides.
Use 1: a present plate unread for scene_check_after_s is looked at; two looks
with no vehicle confirm the departure early, two with a vehicle hold the stay
open (looking again every scene_relook_every_s) until the plate is read again
or the vehicle is gone. Use 2: with no car present a look every
scene_idle_every_s; a vehicle reported for scene_unreadable_after_s with no
plate read raises UNREADABLE_VEHICLE, and two looks with no vehicle after that
raise SPOT_CLEAR.
"""

import base64
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spotter.alerts import http_json
from spotter.events import ArriveLeaveRule

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-8B"
LOOK_WIDTH = 1280
TIMEOUT_S = 15.0
MAX_WORDS = 12
MAX_TOKENS = 120
UNREADABLE_VEHICLE, SPOT_CLEAR = "UNREADABLE_VEHICLE", "SPOT_CLEAR"

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_in_driveway": {"type": "boolean"},
        "plate_visible": {"type": "boolean"},
        "what": {"type": "string", "maxLength": 100},
    },
    "required": ["vehicle_in_driveway", "plate_visible", "what"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Report:
    vehicle_in_driveway: bool
    plate_visible: bool
    what: str  # twelve words at most
    latency_s: float = 0.0


def question(spot: str) -> str:
    return (
        f"This is a fixed camera view. The parking spot is {spot}. Look only at that spot and ignore "
        "any vehicle on the street or beyond it in the background. Answer in JSON: vehicle_in_driveway "
        "(true if a car, van, truck or motorcycle is in the spot), plate_visible (true if a licence "
        "plate on that vehicle can be seen), what (at most twelve words describing the vehicle or "
        "the empty spot, for example \"a white SUV facing the camera\")."
    )


def shrink_to_data_url(frame: np.ndarray, width: int = LOOK_WIDTH) -> str:
    height, current = frame.shape[:2]
    if current > width:
        frame = cv2.resize(frame, (width, round(height * width / current)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the frame")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def parse_report(text: str, latency_s: float = 0.0) -> Report | None:
    """A Report from the model's JSON, or None when it is not the JSON asked for."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    vehicle, plate, what = data.get("vehicle_in_driveway"), data.get("plate_visible"), data.get("what")
    if not isinstance(vehicle, bool) or not isinstance(plate, bool) or not isinstance(what, str):
        return None
    words = what.strip().split()
    return Report(vehicle, plate, " ".join(words[:MAX_WORDS]), latency_s)


class CosmosScene:
    """Asks the Cosmos endpoint about one frame; None when it cannot answer in time."""

    def __init__(
        self,
        url: str,
        token: str,
        spot: str,
        model: str = DEFAULT_MODEL,
        timeout_s: float = TIMEOUT_S,
        post: Callable[[str, bytes, str, dict, float], dict] | None = None,
        say: Callable[[str], None] = print,
    ) -> None:
        self.url = url.rstrip("/")
        self._token = token
        self.question = question(spot)
        self.model = model
        self.timeout_s = timeout_s
        self._post = post or self._http_post
        self.say = say
        self.last_error: str | None = None

    def _http_post(self, url: str, body: bytes, content_type: str, headers: dict, timeout_s: float) -> dict:
        import urllib.request

        request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type, **headers}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())

    def look(self, frame: np.ndarray) -> Report | None:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": shrink_to_data_url(frame)}},
                        {"type": "text", "text": self.question},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "scene_report", "schema": REPORT_SCHEMA, "strict": True},
            },
        }
        started = time.perf_counter()
        try:
            reply = self._post(
                f"{self.url}/v1/chat/completions",
                json.dumps(body).encode(),
                "application/json",
                {"Authorization": f"Bearer {self._token}"},
                self.timeout_s,
            )
            content = reply["choices"][0]["message"].get("content") or ""
        except Exception as err:  # timeouts, HTTP errors, malformed replies
            self.last_error = f"{type(err).__name__}: {err}".replace(self._token, "[REDACTED]")[:200]
            return None
        report = parse_report(content, time.perf_counter() - started)
        if report is None:
            self.last_error = f"not the JSON asked for: {content[:120]!r}"
        return report


class ScriptedScene:
    """Canned reports for tests, from a JSON file: a list of entries, each applying until its
    "until" time on the rule clock (seconds since the run started); the last entry applies
    forever. An entry with "unavailable": true answers nothing.

        [{"until": 10, "vehicle_in_driveway": false, "plate_visible": false, "what": "empty asphalt"},
         {"until": 22, "vehicle_in_driveway": true, "plate_visible": true, "what": "a white SUV"},
         {"vehicle_in_driveway": false, "plate_visible": false, "what": "empty asphalt"}]
    """

    def __init__(self, path: str | Path) -> None:
        self.entries = json.loads(Path(path).read_text())
        if not isinstance(self.entries, list) or not self.entries:
            raise ValueError(f"{path}: expected a non-empty list of reports")
        self.last_error: str | None = None

    def look(self, frame: np.ndarray, now: float = 0.0) -> Report | None:
        entry = next((e for e in self.entries if "until" not in e or now < e["until"]), self.entries[-1])
        if entry.get("unavailable"):
            self.last_error = "scripted: unavailable"
            return None
        return Report(bool(entry["vehicle_in_driveway"]), bool(entry.get("plate_visible", False)), str(entry.get("what", "")))


@dataclass(frozen=True)
class Look:
    """One finished look: when it was asked for (rule clock) and what came back."""

    tag: str  # "departure" or "idle"
    asked_at: float
    report: Report | None  # None: scene unavailable


class SceneWorker:
    """Runs looks on a thread of its own and keeps only the newest request."""

    def __init__(self, scene, say: Callable[[str], None] = print) -> None:
        self.scene = scene
        self.say = say
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self.looks = 0
        self.unavailable = 0
        self.latencies: list[float] = []
        self._thread = threading.Thread(target=self._run, name="scene-looks", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending > 0

    def request(self, frame: np.ndarray, now: float, tag: str) -> bool:
        """Ask for a look at this frame; False if one is already in flight."""
        with self._lock:
            if self._pending:
                return False
            self._pending += 1
        self._requests.put((frame.copy(), now, tag))
        return True

    def results(self) -> list[Look]:
        done = []
        while True:
            try:
                done.append(self._results.get_nowait())
            except queue.Empty:
                return done

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                return
            frame, now, tag = item
            try:
                started = time.perf_counter()
                if isinstance(self.scene, ScriptedScene):
                    report = self.scene.look(frame, now)
                else:
                    report = self.scene.look(frame)
                self.looks += 1
                if report is None:
                    self.unavailable += 1
                    self.say(f"SCENE look ({tag}) unavailable: {getattr(self.scene, 'last_error', None) or 'no answer'}")
                else:
                    self.latencies.append(report.latency_s or (time.perf_counter() - started))
                    self.say(
                        f"SCENE look ({tag}): vehicle {'yes' if report.vehicle_in_driveway else 'no'},"
                        f" plate {'visible' if report.plate_visible else 'not visible'}, {report.what!r}"
                        + (f", {report.latency_s:.1f}s" if report.latency_s else "")
                    )
                self._results.put(Look(tag, now, report))
            finally:
                with self._lock:
                    self._pending -= 1

    def stop(self) -> None:
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)


@dataclass
class SceneSettings:
    check_after_s: float = 8.0  # a present plate unread this long gets a look
    relook_every_s: float = 15.0  # while the scene still sees the car
    idle_every_s: float = 15.0  # looks while no car is present
    unreadable_after_s: float = 20.0  # a vehicle seen this long with no plate read


@dataclass
class Decision:
    kind: str  # LEFT_CONFIRMED, UNREADABLE_VEHICLE, SPOT_CLEAR
    plate: str = ""
    what: str = ""
    since: float = 0.0  # UNREADABLE_VEHICLE / SPOT_CLEAR: when the vehicle was first reported
    until: float = 0.0  # SPOT_CLEAR: when it was last reported


@dataclass
class Agreement:
    """Consecutive looks that agree on vehicle_in_driveway; an unavailable look resets it."""

    value: bool | None = None
    count: int = 0
    first_at: float = 0.0
    last_at: float = 0.0

    def add(self, look: Look) -> None:
        if look.report is None:
            self.value, self.count = None, 0
            return
        if look.report.vehicle_in_driveway == self.value:
            self.count += 1
        else:
            self.value, self.count, self.first_at = look.report.vehicle_in_driveway, 1, look.asked_at
        self.last_at = look.asked_at

    def agreed(self, value: bool) -> bool:
        return self.value is value and self.count >= 2

    def reset(self) -> None:
        self.value, self.count = None, 0


class SceneWatch:
    """Decides, from looks, when a departure is confirmed or doubted and when an unreadable
    vehicle has come and gone. Nothing acts on one look."""

    def __init__(self, worker: SceneWorker, rule: ArriveLeaveRule, settings: SceneSettings, say: Callable[[str], None] = print) -> None:
        self.worker = worker
        self.rule = rule
        self.settings = settings
        self.say = say
        self.agreement = Agreement()
        self.next_look_at: float | None = None  # rule time of the next scheduled look, if any
        self.vehicle_since: float | None = None  # idle mode: when a vehicle was first reported in this run
        self.vehicle_until: float | None = None
        self.unreadable_raised = False
        self.last_report: Report | None = None
        self.mode = "idle"  # "idle" while no car is present, "departure" while one is

    def _ask(self, frame: np.ndarray, now: float, tag: str) -> None:
        if self.worker.request(frame, now, tag):
            self.next_look_at = None

    def tick(self, now: float, frame: np.ndarray, plates_read: set[str], last_read: dict[str, float]) -> list[Decision]:
        """One checked frame. `last_read` maps each present plate to the rule time of its last read."""
        decisions: list[Decision] = []
        for look in self.worker.results():
            self.agreement.add(look)
            if look.report is not None:
                self.last_report = look.report
            if look.report is None and self.rule.held:
                # Scene unavailable: the leave_after_s rule applies as if the scene were off.
                for plate in sorted(self.rule.held):
                    self.rule.release(plate)
                    self.say(f"SCENE unavailable: the hold on a stay is released, leave_after_s applies")
            if look.tag == "departure":
                decisions += self._departure_results(now, look)
            else:
                decisions += self._idle_results(now, look)

        present = set(self.rule.present)
        mode = "departure" if present else "idle"
        if mode != self.mode:
            # A car arrived or the spot emptied: the other mode's schedule and looks no longer apply.
            self.mode = mode
            self.next_look_at = None
            self.agreement.reset()
        if present:
            self.vehicle_since = self.vehicle_until = None  # a known car is here; the idle watch is off
            self.unreadable_raised = False
            # Plates read again release their hold.
            for plate in sorted(self.rule.held & plates_read):
                self.rule.release(plate)
                self.agreement.reset()
                self.next_look_at = None
                self.say(f"SCENE: the plate was read again, the stay is no longer held")
            unread = [p for p in present if now - last_read.get(p, now) >= self.settings.check_after_s]
            if unread and not self.worker.pending:
                if self.rule.held & set(unread):
                    if self.next_look_at is None or now >= self.next_look_at:
                        self._ask(frame, now, "departure")
                elif self.next_look_at is None or now >= self.next_look_at:
                    self._ask(frame, now, "departure")
        else:
            if self.next_look_at is None or now >= self.next_look_at:
                if not self.worker.pending:
                    self._ask(frame, now, "idle")
        return decisions

    def _departure_results(self, now: float, look: Look) -> list[Decision]:
        decisions = []
        unread = sorted(self.rule.present)
        if not unread:
            return decisions
        if self.agreement.agreed(False):
            for plate in unread:
                decisions.append(Decision("LEFT_CONFIRMED", plate=plate))
            self.agreement.reset()
            self.next_look_at = None
        elif self.agreement.agreed(True):
            for plate in unread:
                if plate not in self.rule.held:
                    self.rule.hold(plate)
                    self.say("SCENE: two looks see the vehicle; the stay is held open past leave_after_s")
            self.next_look_at = now + self.settings.relook_every_s
        elif look.report is not None:
            self.next_look_at = now  # one look so far: look again at once
        else:
            self.next_look_at = now + self.settings.relook_every_s if look.report is None else now
        return decisions

    def _idle_results(self, now: float, look: Look) -> list[Decision]:
        decisions = []
        if self.agreement.agreed(True):
            if self.vehicle_since is None:
                self.vehicle_since = self.agreement.first_at
            self.vehicle_until = look.asked_at
            if not self.unreadable_raised and self.vehicle_until - self.vehicle_since >= self.settings.unreadable_after_s:
                self.unreadable_raised = True
                what = look.report.what if look.report else ""
                decisions.append(Decision(UNREADABLE_VEHICLE, what=what, since=self.vehicle_since))
        elif self.agreement.agreed(False):
            if self.unreadable_raised:
                decisions.append(Decision(SPOT_CLEAR, since=self.vehicle_since or now, until=self.vehicle_until or now))
                self.unreadable_raised = False
            if self.agreement.count >= 2:
                self.vehicle_since = self.vehicle_until = None
        self.next_look_at = now + self.settings.idle_every_s
        return decisions


def make_scene(setting: str, env: dict, spot: str, timeout_s: float = TIMEOUT_S, say: Callable[[str], None] = print):
    """The scene named by SPOTTER_SCENE: None for off, a CosmosScene, or a ScriptedScene."""
    if setting == "off":
        return None
    if setting == "cosmos":
        url = (env.get("COSMOS_URL") or "").strip()
        token = (env.get("COSMOS_TOKEN") or "").strip()
        if not url or not token:
            raise ValueError("SPOTTER_SCENE=cosmos needs COSMOS_URL and COSMOS_TOKEN in .env (scripts/cosmos_up.py writes them)")
        if not spot:
            raise ValueError("SPOTTER_SCENE=cosmos needs SPOTTER_SPOT_DESCRIPTION in .env")
        return CosmosScene(url, token, spot, timeout_s=timeout_s, say=say)
    if setting.startswith("script:"):
        return ScriptedScene(setting[len("script:"):])
    raise ValueError(f"SPOTTER_SCENE must be off, cosmos or script:<file>, got {setting!r}")
