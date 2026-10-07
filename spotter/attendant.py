"""The attendant: Nemotron decides what to do about each event by calling the tools.

For every ARRIVED or LEFT event the model gets a short system prompt, the event
as JSON and the six tools, and is called again with each tool's result until it
stops calling tools, for at most MAX_ROUNDS rounds. Every tool call and result
goes to action_log. If a model call fails twice, that is logged and the owner is
alerted that the attendant is offline.
"""

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from spotter.alerts import Alert, Alerts
from spotter.db import Database, iso, parse
from spotter.events import Event
from spotter.light import Light
from spotter.privacy import Masker
from spotter.tools import (
    DEFAULT_MIN_CHARGE_CENTS,
    DEFAULT_OVERSTAY_FEE_CENTS,
    TOOL_SCHEMAS,
    EventContext,
    Toolbox,
    dollars,
)

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "nvidia/Nemotron-3_5-Lightning"
MAX_ROUNDS = 10
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
4. ENDING_SOON: send_alert to the driver saying how many minutes are left (minutes_left in the event).
5. OVERSTAY: set_light amber, send_alert to the driver that the booking has ended and an overstay fee now applies, and send_alert to the owner that the car is overstaying.
6. LEFT with an open stay: end_session, compute_bill, send_alert the bill to the driver, giving the breakdown (parking, overstay fee, total) when a fee applied, then set_light off.
7. LEFT with no open stay: send_alert to the owner that the unbooked car has gone and how long it stayed (stayed_for in the event), then set_light off.
8. When done, reply with one short sentence and no more tool calls.
"""


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

    def handle(self, event: Event, snapshot: str, at: datetime) -> None:
        """Let the model act on one event, printing everything it does."""
        self.stats.events += 1
        self.masker.for_event(event.plate)
        say = self.masker.print
        context = EventContext(event.type, event.plate, at, snapshot, time.perf_counter())
        toolbox = Toolbox(
            self.db, self.light, self.alerts, context, self.min_charge_cents, self.overstay_fee_cents
        )
        payload = {
            "event": event.type,
            "plate": event.plate,
            "time": iso(at),
            "reads": event.reads,
            "snapshot": snapshot,
        }
        if event.type == "LEFT":
            payload["stayed_for"] = describe_duration(event.last_seen - event.first_seen)
        elif event.type in ("ENDING_SOON", "OVERSTAY"):
            payload.update(self._booking_timing(event.plate, at, event.type))
        logged_event = {"type": event.type, "plate": event.plate, "time": iso(at)}
        say(f"=== {event.type} {self.masker.label(event.plate)} at {iso(at)} ===")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        for round_number in range(1, self.max_rounds + 1):
            reply = self._ask(messages)
            if reply is None:
                self._offline(logged_event, snapshot)
                return
            message, usage, latency = reply
            tool_calls = message.get("tool_calls") or []
            say(
                f"MODEL round {round_number} | {latency:.2f}s | {usage_line(usage)}"
                f" | {len(tool_calls)} tool call(s)"
            )
            if not tool_calls:
                final = (message.get("content") or "").strip()
                say(f"ATTENDANT: {final or '(no final sentence)'}")
                self.db.log_action(logged_event, "final", {"round": round_number}, {"text": final})
                return
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
                self.stats.alert_failures += bool(result.get("delivery_error"))
                self.db.log_action(logged_event, name, arguments, result)
                say(f"  -> {json.dumps(result)}")
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                )

        say(f"ATTENDANT: stopped after {self.max_rounds} rounds of tool calls")
        self.stats.rounds_exhausted += 1
        self.db.log_action(
            logged_event, "model", {"rounds": self.max_rounds}, {"error": "stopped: too many rounds"}
        )

    def _booking_timing(self, plate: str, at: datetime, kind: str) -> dict:
        """What a timer event needs the model to know: when the booking ends, and by how much."""
        stay = self.db.open_session(plate)
        booking = self.db.booking(stay["booking_id"]) if stay else None
        if booking is None:
            return {}
        ends_at = parse(booking["ends_at"])
        timing = {"session_id": stay["id"], "booking_ends_at": booking["ends_at"]}
        if kind == "ENDING_SOON":
            seconds_left = max(0.0, (ends_at - at).total_seconds())
            timing["minutes_left"] = math.ceil(seconds_left / 60)
            timing["seconds_left"] = round(seconds_left)
        else:
            timing["ended_ago"] = describe_duration((at - ends_at).total_seconds())
            timing["overstay_fee"] = dollars(self.overstay_fee_cents)
        return timing

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

    def _offline(self, logged_event: dict, snapshot: str) -> None:
        """The model could not be reached twice: log it and tell the owner."""
        self.stats.offline_events += 1
        text = "Spotter attendant is offline: the model call failed twice, so this event was not handled."
        self.db.log_action(
            logged_event, "model", {"attempts": ATTEMPTS}, {"error": "offline: the model call failed twice"}
        )
        error = self.alerts.send(Alert("owner", None, text, snapshot))
        self.stats.alert_failures += error is not None
        self.db.log_action(
            logged_event,
            "send_alert",
            {"to": "owner", "message": text},
            {"sent_to": "owner", "by": "code", "delivered": error is None, "delivery_error": error},
        )

    def _safe(self, err: Exception) -> str:
        return f"{type(err).__name__}: {err}".replace(str(self.client.api_key), "[REDACTED]")

    def print_stats(self) -> None:
        s = self.stats
        print("--- attendant ---")
        print(
            f"events handled: {s.events} | model calls: {s.model_calls} ({s.model_failures} failed)"
            f" | tool calls: {s.tool_calls} ({s.tool_errors} errors returned to the model)"
            f" | alerts not delivered: {s.alert_failures}"
            f" | offline events: {s.offline_events} | rounds exhausted: {s.rounds_exhausted}"
        )
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
