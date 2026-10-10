"""Run the attendant on a video source: watcher events in, tool calls out.

    python -m spotter.run --source samples/IMG_2472.mov [--expect PLATE]          # replay
    python -m spotter.run --source samples/IMG_2472.mov --live [--expect PLATE]   # the file as if it were a camera
    python -m spotter.run --source rtsp://CAMERA/stream --leave-after-s 60        # the camera (or SPOTTER_SOURCE in .env)

The watcher (spotter.watch) finds ARRIVED and LEFT events, the timers add
ENDING_SOON and OVERSTAY, and each one goes to the attendant
(spotter.attendant), one at a time and in order; a replay pauses while the
attendant works. For a replay, video time is added to the moment the run
started, so bookings and bills use real timestamps.

A camera outage is reported here, from code, not by the model: one owner alert
when frames stop and one when they return. At start the open stays from an
earlier run are counted, and the light is set off if there are none.

Needs NEBIUS_API_KEY in .env; TELEGRAM_BOT_TOKEN when alerts go to Telegram;
SPOTTER_LIGHT_HOST (and KASA_USERNAME/KASA_PASSWORD if the bulb asks) when the
light is the Kasa bulb. With --expect PLATE everything printed, tool arguments
and model replies included, says MINE or OTHER instead of plate text.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from openai import OpenAI

from spotter import watch
from spotter.alerts import Alert, TelegramSetupError, make_alerts
from spotter.attendant import BASE_URL, Attendant
from spotter.db import Database, iso
from spotter.events import Event
from spotter.light import LightError, make_light
from spotter.timers import StayTimer
from spotter.words import clock_words, duration_words


def parse_time(text: str) -> datetime:
    """An ISO 8601 time; one without a zone is taken as UTC."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="python -m spotter.run",
        description="Watch a video source and let the Nemotron attendant act on each event.",
    )
    parser.add_argument(
        "--replay-base-time",
        type=parse_time,
        metavar="ISO_TIME",
        help="for a replay, the real time of video second 0 (default: when the run starts)",
    )
    settings, masker, args = watch.parse_args(parser)  # also loads .env
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        print("NEBIUS_API_KEY is missing: add it to .env", file=sys.stderr)
        return 2
    try:
        alerts = make_alerts(settings.alerts, masker.print)
    except TelegramSetupError as err:
        print(f"Telegram alerts cannot start: {err}", file=sys.stderr)
        return 2
    try:
        light = make_light(settings.light, settings.light_host, settings.light_brightness, masker.print)
    except LightError as err:
        print(f"the light cannot start: {err}", file=sys.stderr)
        return 2
    print(
        f"alerts: {settings.alerts} | light: {settings.light}"
        + (f" at {settings.light_host}, brightness {settings.light_brightness}" if settings.light == "kasa" else "")
        + f" | minimum charge: {settings.min_charge_cents} cents"
        f" | ending soon: {settings.ending_soon_s:g}s before the booking ends"
        f" | overstay: {settings.overstay_grace_s:g}s after it, fee {settings.overstay_fee_cents} cents"
    )

    db = Database()
    open_stays = db.open_sessions()
    print(f"stays still open from an earlier run: {len(open_stays)}")
    if not open_stays:
        try:
            light.set("off")
        except LightError as err:
            print(f"the light did not respond: {err}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120, max_retries=0)
    attendant = Attendant(
        client,
        db,
        light,
        alerts,
        masker,
        min_charge_cents=settings.min_charge_cents,
        overstay_fee_cents=settings.overstay_fee_cents,
    )
    timer = StayTimer(db, settings.ending_soon_s, settings.overstay_grace_s)

    def timers(now: float, real_time: datetime, plates: set[str]) -> list[Event]:
        return [Event(t.type, t.plate, now, now, now, 0) for t in timer.check(real_time, plates)]

    def on_camera(kind: str, when: datetime, down: float | None) -> None:
        """Camera outages are handled by code: one plain owner alert each way, and a log entry."""
        if kind == watch.CAMERA_OFFLINE:
            text = f"Camera offline: no frames since {clock_words(when)}. Stays and the light are left as they are."
        else:
            text = f"Camera back after {duration_words(down or 0)}."
        error = alerts.send(Alert("owner", None, text, ""))
        logged_event = {"type": kind, "time": iso(when)}
        db.log_action(logged_event, "camera", {"downtime_s": round(down, 1) if down else None}, {"by": "code"})
        db.log_action(
            logged_event,
            "send_alert",
            {"to": "owner", "message": text, "by": "code"},
            {"sent_to": "owner", "delivered": error is None, "delivery_error": error, "by": "code"},
        )

    try:
        summary = watch.run(
            settings, masker, attendant.handle, timers, args.replay_base_time, live=args.live, on_camera=on_camera
        )
    except (FileNotFoundError, RuntimeError) as err:
        print(err, file=sys.stderr)
        return 2
    finally:
        db.close()
    watch.print_summary(summary, masker)
    attendant.print_stats()
    return 0


if __name__ == "__main__":
    watch.exit_now(main())  # the database is closed inside main()
