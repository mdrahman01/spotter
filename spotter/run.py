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

from openai import OpenAI

from spotter import watch
from spotter.alerts import TelegramSetupError, make_alerts
from spotter.attendant import BASE_URL, Attendant
from spotter.db import Database
from spotter.light import ConsoleLight


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="python -m spotter.run",
        description="Watch a video source and let the Nemotron attendant act on each event.",
    )
    settings, masker = watch.parse_args(parser)  # also loads .env
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        print("NEBIUS_API_KEY is missing: add it to .env", file=sys.stderr)
        return 2
    try:
        alerts = make_alerts(settings.alerts, masker.print)
    except TelegramSetupError as err:
        print(f"Telegram alerts cannot start: {err}", file=sys.stderr)
        return 2
    print(f"alerts: {settings.alerts} | minimum charge: {settings.min_charge_cents} cents")

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120, max_retries=0)
    attendant = Attendant(
        client,
        Database(),
        ConsoleLight(say=masker.print),
        alerts,
        masker,
        min_charge_cents=settings.min_charge_cents,
    )
    try:
        summary = watch.run(settings, masker, attendant.handle)
    except (FileNotFoundError, RuntimeError) as err:
        print(err, file=sys.stderr)
        return 2
    watch.print_summary(summary, masker)
    attendant.print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
