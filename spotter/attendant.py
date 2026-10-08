"""The attendant: Nemotron acts on each event by calling the tools, and code finishes the job.

For every event the model gets a short system prompt, the event as JSON and the
six tools, and is called again with each tool's result until it stops calling
tools, for at most MAX_ROUNDS rounds. Every tool call and result goes to
action_log.

Then the finish check runs: from the database, the light and the alerts of
this event, code works out what the event's rules require and what is still
missing. If something is missing, the model is told once, in plain words, what
to do and gets REMINDER_ROUNDS more rounds; whatever is still missing after
that, code does itself and records as done by code. The same check runs when
the model cannot be reached or runs out of rounds, so there is one path.
"""

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from spotter.alerts import Alert, Alerts
from spotter.db import Database, iso, parse
from spotter.events import Event, EventFrames
from spotter.light import Light
from spotter.privacy import Masker
from spotter.timers import time_left_words
from spotter.tools import (
    DEFAULT_MIN_CHARGE_CENTS,
    DEFAULT_OVERSTAY_FEE_CENTS,
    TOOL_SCHEMAS,
    EventContext,
    Toolbox,
    bill_for_session,
    dollars,
)

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "nvidia/Nemotron-3_5-Lightning"
MAX_ROUNDS = 10
REMINDER_ROUNDS = 3  # rounds the model gets after being told what is missing
ATTEMPTS = 2  # tries per model call before the attendant counts as offline

SYSTEM_PROMPT = """\
You are Spotter, the automated attendant for one private driveway.
Each user message is one event (ARRIVED, ENDING_SOON, OVERSTAY or LEFT) for one licence plate, as JSON.
You act only through the tools. The code behind them enforces the rules and does
all arithmetic; a tool that returns {"error": ...} did nothing.

Rules:
1. Always call lookup_booking first. Never assume a booking.
2. ARRIVED with a booking whose status is "active": start_session, set_light green, send_alert to the driver.
3. ARRIVED with anything else (no booking, or outside its window): set_light red, then send_alert to the owner saying that an unbooked car is in the spot and that a photo is attached. Do not suggest contacting the driver.
4. ENDING_SOON: send_alert to the driver saying how much time is left, in the words of time_left in the event.
5. OVERSTAY: set_light amber, send_alert to the driver that the booking has ended and an overstay fee of overstay_fee (in the event) now applies, and send_alert to the owner that the car is overstaying.
6. LEFT with an open stay: end_session, compute_bill, send_alert the bill to the driver, giving the breakdown (parking, overstay fee, total) when the bill has an overstay fee; if the stay had overstayed, also send_alert to the owner that the car has gone and what was billed; then set_light off.
7. LEFT with no open stay: send_alert to the owner that the car has gone and how long it stayed (stayed_for in the event), then set_light off.
8. When done, reply with one short sentence listing what you did, and make no more tool calls.
"""

CLEAN, AFTER_REMINDER, BY_CODE = "clean", "after a reminder", "completed by code"


@dataclass
class Stats:
    events: int = 0
    model_calls: int = 0
    model_failures: int = 0
    latencies: list[float] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    reasoning_tokens: list[int] = field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    alert_failures: int = 0
    offline_events: int = 0
    rounds_exhausted: int = 0
    reminders: int = 0
    code_items: int = 0
    undeliverable: int = 0
    finished: Counter = field(default_factory=Counter)  # CLEAN / AFTER_REMINDER / BY_CODE
    finish_lines: list[str] = field(default_factory=list)  # one per event, for the summary


@dataclass(frozen=True)
class Expected:
    """Facts fixed when the event arrives; the required end state follows from them."""

    kind: str
    plate: str
    booking: dict | None  # the booking active at the event's time, if any
    stay: dict | None  # the stay open when the event arrived, if any
    time_left: str | None = None  # ENDING_SOON: the words made by code
    stayed_for: str | None = None  # LEFT of an unbooked car

    @property
    def overstayed(self) -> bool:
        return bool(self.stay and self.stay["overstayed_at"])


@dataclass
class Missing:
    """One thing the event's rules require that has not happened."""

    key: str
    tell: str  # plain words for the model
    fix: Callable[[], None] | None  # what code does about it; None when code cannot


def describe_duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} seconds"


class Attendant:
    def __init__(
        self,
        client: OpenAI,
        db: Database,
        light: Light,
        alerts: Alerts,
        masker: Masker,
        model: str = MODEL,
        max_rounds: int = MAX_ROUNDS,
        min_charge_cents: int = DEFAULT_MIN_CHARGE_CENTS,
        overstay_fee_cents: int = DEFAULT_OVERSTAY_FEE_CENTS,
    ) -> None:
        self.client = client
        self.db = db
        self.light = light
        self.alerts = alerts
        self.masker = masker
        self.model = model
        self.max_rounds = max_rounds
        self.min_charge_cents = min_charge_cents
        self.overstay_fee_cents = overstay_fee_cents
        self.stats = Stats()
        self._alerts: list[tuple[str, str, bool]] = []  # (to, message, delivered) in this event
        self._logged_event: dict = {}

    # ----------------------------------------------------------------- one event

    def handle(self, event: Event, frames: EventFrames) -> None:
        """Let the model act on one event, then make sure the event's rules were met."""
        self.stats.events += 1
        self.masker.for_event(event.plate)
        say = self.masker.print
        at = frames.at
        left = event.type == "LEFT"
        context = EventContext(
            event.type,
            event.plate,
            at,
            frames.snapshot,
            last_read_at=frames.last_read_at if left else None,
            last_read_snapshot=frames.last_read_snapshot if left else None,
            started=time.perf_counter(),
        )
        toolbox = Toolbox(
            self.db, self.light, self.alerts, context, self.min_charge_cents, self.overstay_fee_cents
        )
        expected = self._expected(event, context)
        self._alerts = []
        self._logged_event = {"type": event.type, "plate": event.plate, "time": iso(at)}
        if left:
            self._logged_event["last_read_at"] = iso(frames.last_read_at)

        # The model gets facts it acts on, not file paths: the code attaches snapshots.
        payload = {"event": event.type, "plate": event.plate, "time": iso(at), "reads": event.reads}
        if left:
            payload["last_read_at"] = iso(frames.last_read_at)  # the car left when it was last read
            if expected.stay is None:
                payload["stayed_for"] = expected.stayed_for
        elif event.type == "ENDING_SOON" and expected.stay:
            payload["booking_ends_at"] = self.db.booking(expected.stay["booking_id"])["ends_at"]
            payload["time_left"] = expected.time_left
        elif event.type == "OVERSTAY" and expected.stay:
            ends_at = self.db.booking(expected.stay["booking_id"])["ends_at"]
            payload["booking_ended_at"] = ends_at
            payload["ended_ago"] = describe_duration((at - parse(ends_at)).total_seconds())
            payload["overstay_fee"] = dollars(self.overstay_fee_cents)
        say(f"=== {event.type} {self.masker.label(event.plate)} at {iso(at)} ===")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        outcome = self._converse(messages, toolbox, self.max_rounds)
        missing = self._missing(expected, context)
        how = CLEAN
        reminded_about: list[str] = []
        if missing and outcome != "offline":
            reminded_about = [item.tell for item in missing]
            self._remind(messages, event.type, missing)
            self._converse(messages, toolbox, REMINDER_ROUNDS)
            missing = self._missing(expected, context)
            how = AFTER_REMINDER
        done_by_code: list[str] = []
        if missing:
            done_by_code = [item.tell for item in missing]
            self._complete(missing)
            how = BY_CODE
        self.stats.finished[how] += 1
        self._record(
            "finish",
            {"how": how},
            {"reminded_about": reminded_about, "completed_by_code": done_by_code, "model": outcome},
            quiet=True,
        )
        detail = ""
        if how == AFTER_REMINDER:
            detail = f" about: {'; '.join(reminded_about)}"
        elif how == BY_CODE:
            detail = f": {'; '.join(done_by_code)}"
        say(f"FINISH: {how}{detail}")
        self.stats.finish_lines.append(f"{event.type} {how}{detail}")

    def _expected(self, event: Event, context: EventContext) -> Expected:
        stay = self.db.open_session(event.plate)
        booking = self.db.booking_at(event.plate, context.at)
        time_left = None
        if event.type == "ENDING_SOON" and stay:
            ends_at = parse(self.db.booking(stay["booking_id"])["ends_at"])
            time_left = time_left_words(max(0.0, (ends_at - context.at).total_seconds()))
        stayed_for = None
        if event.type == "LEFT" and stay is None:
            stayed_for = describe_duration(event.last_seen - event.first_seen)
        return Expected(event.type, event.plate, booking, stay, time_left, stayed_for)

    # ------------------------------------------------------------ the model loop

    def _converse(self, messages: list[dict], toolbox: Toolbox, rounds: int) -> str:
        """Call the model until it stops calling tools: "final", "rounds" or "offline"."""
        say = self.masker.print
        for round_number in range(1, rounds + 1):
            reply = self._ask(messages)
            if reply is None:
                self.stats.offline_events += 1
                self._record(
                    "model", {"attempts": ATTEMPTS}, {"error": "offline: the model call failed twice"}, quiet=True
                )
                return "offline"
            message, usage, latency = reply
            tool_calls = message.get("tool_calls") or []
            say(
                f"MODEL round {round_number} | {latency:.2f}s | {usage_line(usage)}"
                f" | {len(tool_calls)} tool call(s)"
            )
            if not tool_calls:
                final = (message.get("content") or "").strip()
                messages.append({"role": "assistant", "content": final})
                say(f"ATTENDANT: {final or '(no final sentence)'}")
                self._record("final", {"round": round_number}, {"text": final}, quiet=True)
                return "final"
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"] or "{}"
                say(f"TOOL {name} {arguments}")
                result = toolbox.call(name, arguments)
                self.stats.tool_calls += 1
                self.stats.tool_errors += "error" in result
                if name == "send_alert" and "error" not in result:
                    self._note_alert(arguments, result)
                self.db.log_action(self._logged_event, name, arguments, result)
                say(f"  -> {json.dumps(result)}")
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                )
        say(f"ATTENDANT: stopped after {rounds} rounds of tool calls")
        self.stats.rounds_exhausted += 1
        self._record("model", {"rounds": rounds}, {"error": "stopped: too many rounds"}, quiet=True)
        return "rounds"

    def _note_alert(self, arguments: str, result: dict) -> None:
        try:
            message = json.loads(arguments).get("message") or ""
        except (ValueError, AttributeError):
            message = ""
        delivered = bool(result.get("delivered"))
        self.stats.alert_failures += not delivered
        self._alerts.append((result.get("sent_to"), message, delivered))

    def _ask(self, messages: list[dict]) -> tuple[dict, dict, float] | None:
        """One model call, tried up to ATTEMPTS times: (message, usage, latency) or None."""
        for attempt in range(1, ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model, messages=messages, tools=TOOL_SCHEMAS, temperature=0
                )
                body = json.loads(raw.text)
                message = body["choices"][0]["message"]
            except Exception as err:  # API errors, timeouts, malformed replies
                self.stats.model_failures += 1
                self.masker.print(
                    f"MODEL call failed (attempt {attempt} of {ATTEMPTS}): {self._safe(err)}"
                )
                continue
            latency = time.perf_counter() - start
            usage = body.get("usage") or {}
            self.stats.model_calls += 1
            self.stats.latencies.append(latency)
            self.stats.completion_tokens.append(usage.get("completion_tokens") or 0)
            self.stats.reasoning_tokens.append(reasoning_tokens(usage) or 0)
            return message, usage, latency
        return None

    def _safe(self, err: Exception) -> str:
        return f"{type(err).__name__}: {err}".replace(str(self.client.api_key), "[REDACTED]")

    # ------------------------------------------------------------- finish check

    def _delivered(self, to: str, needle: str | None = None) -> bool:
        """Was an alert to `to` delivered during this event, containing `needle` if given?"""
        return any(
            delivered and sent_to == to and (needle is None or needle in message)
            for sent_to, message, delivered in self._alerts
        )

    def _missing(self, expected: Expected, context: EventContext) -> list[Missing]:
        """What the event's rules require that has not happened, in the order code would do it."""
        light = self.light.color
        plate = expected.plate
        items: list[Missing] = []

        def want_light(color: str) -> None:
            if light != color:
                items.append(Missing("light", f"set the light to {color}", lambda: self._fix_light(color, context)))

        def want_alert(to: str, tell: str, needle: str | None, text: Callable[[], str]) -> None:
            if not self._delivered(to, needle):
                items.append(Missing(f"{to}_alert", tell, lambda: self._fix_alert(to, text(), context)))

        if expected.kind == "ARRIVED":
            if expected.booking:
                if self.db.open_session(plate) is None:
                    items.append(
                        Missing("stay", "start the stay with start_session", lambda: self._fix_start(expected, context))
                    )
                want_light("green")
                want_alert(
                    "driver",
                    "send the driver an alert that the stay has started",
                    None,
                    lambda: "Your booking is active and your stay has started.",
                )
            else:
                if self.db.open_session(plate) is not None:
                    items.append(Missing("no_stay", "a car without a booking must not have a stay", None))
                want_light("red")
                want_alert(
                    "owner",
                    "send the owner an alert that an unbooked car is in the spot, with the photo attached",
                    None,
                    lambda: "An unbooked car is in the spot. Photo attached.",
                )
        elif expected.kind == "ENDING_SOON":
            words = expected.time_left or "little time"
            want_alert(
                "driver",
                f"send the driver an alert saying the booking ends in {words}",
                expected.time_left,
                lambda: f"Your booking ends in {words}.",
            )
        elif expected.kind == "OVERSTAY":
            fee = dollars(self.overstay_fee_cents)
            want_light("amber")
            want_alert(
                "driver",
                f"send the driver an alert that the booking has ended and an overstay fee of {fee} now applies",
                fee,
                lambda: f"Your booking has ended and an overstay fee of {fee} now applies.",
            )
            want_alert(
                "owner",
                "send the owner an alert that the car is overstaying",
                None,
                lambda: "The car is overstaying.",
            )
        elif expected.kind == "LEFT":
            if expected.stay:
                session = self.db.session(expected.stay["id"])
                if session["status"] != "closed":
                    items.append(Missing("stay_closed", "end the stay with end_session", lambda: self._fix_close(expected, context)))
                if session["amount_cents"] is None:
                    items.append(Missing("bill", "compute the bill with compute_bill", lambda: self._fix_bill(expected)))
                total = dollars(session["amount_cents"]) if session["amount_cents"] is not None else None
                want_alert(
                    "driver",
                    "send the driver an alert that gives the bill total",
                    total or "\0",  # no bill yet: nothing can have given its total
                    lambda: self._bill_text(expected, for_owner=False),
                )
                if expected.overstayed:
                    want_alert(
                        "owner",
                        "send the owner an alert that the car has gone and what was billed",
                        total or "\0",
                        lambda: self._bill_text(expected, for_owner=True),
                    )
            else:
                stayed = expected.stayed_for or "a while"
                want_alert(
                    "owner",
                    "send the owner an alert that the car has gone and how long it stayed",
                    None,
                    lambda: f"The car has gone after {stayed}.",
                )
            want_light("off")
        return items

    def _remind(self, messages: list[dict], kind: str, missing: list[Missing]) -> None:
        """Tell the model once, in plain words, what is still missing and to do only that."""
        text = (
            f"Finish check: for this {kind} event the following is still missing: "
            + "; ".join(item.tell for item in missing)
            + ". Do only that, using the tools, then reply with one short sentence."
        )
        messages.append({"role": "user", "content": text})
        self.stats.reminders += 1
        self.masker.print(f"REMINDER: {text}")
        self._record(
            "reminder", {"missing": [item.key for item in missing]}, {"told": text, "rounds": REMINDER_ROUNDS}, quiet=True
        )

    def _complete(self, missing: list[Missing]) -> None:
        """Do what the model left undone, once each; record what code cannot do."""
        for item in missing:
            if item.fix is None:
                self._record(item.key, {"by": "code"}, {"error": "code cannot complete this: " + item.tell})
                continue
            self.stats.code_items += 1
            item.fix()

    # --- the code fallbacks, each recorded as done by code

    def _record(self, tool: str, arguments: dict, result: dict, quiet: bool = False) -> None:
        self.db.log_action(self._logged_event, tool, arguments, result)
        if not quiet:
            self.masker.print(f"CODE {tool} {json.dumps(arguments)} -> {json.dumps(result)}")

    def _fix_light(self, color: str, context: EventContext) -> None:
        self.light.set(color)
        seconds = round(time.perf_counter() - context.started, 1)
        self._record(
            "set_light", {"color": color, "by": "code"}, {"light": color, "seconds_after_event": seconds, "by": "code"}
        )

    def _fix_alert(self, to: str, text: str, context: EventContext) -> None:
        name = None
        if to == "driver":
            booking = self.db.latest_booking(context.plate)
            name = booking["driver"] if booking else None
        error = self.alerts.send(Alert(to, name, text, context.attachment))
        delivered = error is None
        self.stats.undeliverable += not delivered
        self._alerts.append((to, text, delivered))
        self._record(
            "send_alert",
            {"to": to, "message": text, "by": "code"},
            {"sent_to": to, "name": name, "snapshot": context.attachment, "delivered": delivered,
             "delivery_error": error, "by": "code"},
        )

    def _fix_start(self, expected: Expected, context: EventContext) -> None:
        session_id = self.db.start_session(expected.plate, expected.booking["id"], context.at)
        self._record(
            "start_session",
            {"by": "code"},
            {"session_id": session_id, "booking_id": expected.booking["id"], "arrived_at": iso(context.at), "by": "code"},
        )

    def _fix_close(self, expected: Expected, context: EventContext) -> None:
        self.db.close_session(expected.stay["id"], context.departed_at, context.attachment)
        self._record(
            "end_session",
            {"by": "code"},
            {"session_id": expected.stay["id"], "left_at": iso(context.departed_at), "status": "closed", "by": "code"},
        )

    def _fix_bill(self, expected: Expected) -> None:
        session = self.db.session(expected.stay["id"])
        if session["status"] != "closed":
            self._record("compute_bill", {"by": "code"}, {"error": "the stay is still open", "by": "code"})
            return
        result = bill_for_session(self.db, session, self.min_charge_cents, self.overstay_fee_cents)
        self._record("compute_bill", {"by": "code"}, {**result, "by": "code"})

    def _bill_text(self, expected: Expected, for_owner: bool) -> str:
        """A plain message about the bill, from the stored facts."""
        session = self.db.session(expected.stay["id"])
        if session["amount_cents"] is None:
            return "The car has gone; the bill could not be computed." if for_owner else "Your stay has ended."
        total = dollars(session["amount_cents"])
        if for_owner:
            return f"The car has gone and was billed {total}."
        if session["overstayed_at"]:
            fee = dollars(self.overstay_fee_cents)
            parking = dollars(session["amount_cents"] - self.overstay_fee_cents)
            return f"Your stay has ended. The bill is {total}: parking {parking}, overstay fee {fee}."
        return f"Your stay has ended. The bill is {total}."

    # ---------------------------------------------------------------- summary

    def print_stats(self) -> None:
        s = self.stats
        print("--- attendant ---")
        print(
            f"events handled: {s.events} | model calls: {s.model_calls} ({s.model_failures} failed)"
            f" | tool calls: {s.tool_calls} ({s.tool_errors} errors returned to the model)"
            f" | alerts not delivered: {s.alert_failures + s.undeliverable}"
            f" | offline events: {s.offline_events} | rounds exhausted: {s.rounds_exhausted}"
        )
        print(
            f"finish: {s.finished[CLEAN]} clean, {s.finished[AFTER_REMINDER]} after a reminder,"
            f" {s.finished[BY_CODE]} completed by code | reminders sent: {s.reminders}"
            f" | items done by code: {s.code_items} | alerts code could not deliver: {s.undeliverable}"
        )
        for number, line in enumerate(s.finish_lines, 1):
            print(f"  event {number}: {line}")
        if s.latencies:
            print(
                f"model latency: avg {sum(s.latencies) / len(s.latencies):.2f}s,"
                f" max {max(s.latencies):.2f}s | completion tokens: total {sum(s.completion_tokens)}"
                f" | reasoning tokens: total {sum(s.reasoning_tokens)},"
                f" max {max(s.reasoning_tokens)} in one call"
            )


def reasoning_tokens(usage: dict) -> int | None:
    details = usage.get("completion_tokens_details") or {}
    return details.get("reasoning_tokens")


def usage_line(usage: dict) -> str:
    reasoning = reasoning_tokens(usage)
    return (
        f"tokens: prompt {usage.get('prompt_tokens')}, completion {usage.get('completion_tokens')}"
        + (f" (reasoning {reasoning})" if reasoning is not None else "")
    )
