"""Run the attendant on a video source: watcher events in, tool calls out.

    python -m spotter.run --source samples/IMG_2472.mov [--expect PLATE]
    SPOTTER_ALERTS=telegram python -m spotter.run --source rtsp://CAMERA/stream --leave-after-s 60

The watcher (spotter.watch) finds ARRIVED and LEFT events and hands each one
to the attendant (spotter.attendant), one at a time and in order; a file
replay pauses while the attendant works. For a file, video time is added to
the moment the run started, so bookings and bills use real timestamps.

Needs NEBIUS_API_KEY in .env, and TELEGRAM_BOT_TOKEN when alerts go to
Telegram. The light is a console stand-in that prints. With --expect PLATE
everything printed, tool arguments and model replies included, says MINE or
OTHER instead of plate text.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from openai import OpenAI

from spotter import watch
from spotter.alerts import TelegramSetupError, make_alerts
from spotter.attendant import BASE_URL, Attendant
from spotter.db import Database
from spotter.events import Event
from spotter.light import ConsoleLight
from spotter.timers import StayTimer


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
        help="for a file source, the real time of video second 0 (default: when the run starts)",
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
    print(
        f"alerts: {settings.alerts} | minimum charge: {settings.min_charge_cents} cents"
        f" | ending soon: {settings.ending_soon_s:g}s before the booking ends"
        f" | overstay: {settings.overstay_grace_s:g}s after it, fee {settings.overstay_fee_cents} cents"
    )

    db = Database()
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120, max_retries=0)
    attendant = Attendant(
        client,
        db,
        ConsoleLight(say=masker.print),
        alerts,
        masker,
        min_charge_cents=settings.min_charge_cents,
        overstay_fee_cents=settings.overstay_fee_cents,
    )
    timer = StayTimer(db, settings.ending_soon_s, settings.overstay_grace_s)

    def timers(now: float, real_time: datetime, plates: set[str]) -> list[Event]:
        return [Event(t.type, t.plate, now, now, now, 0) for t in timer.check(real_time, plates)]

    try:
        summary = watch.run(settings, masker, attendant.handle, timers, args.replay_base_time)
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
