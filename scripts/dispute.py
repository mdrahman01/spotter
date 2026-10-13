"""Hand one driver message to the dispute worker without Telegram.

    .venv/bin/python scripts/dispute.py "I left at 6:50, why the fee?" [--expect PLATE] [--alerts console|telegram]
                                        [--waive on|off] [--waive-within-min 10]
    .venv/bin/python scripts/dispute.py --listen 300 --alerts telegram --waive on   # answer texts to the bot for 5 minutes

The message counts as coming from the driver of the stay that ended most
recently. The reply and the owner's note go where SPOTTER_ALERTS (or --alerts)
says; the goodwill rule follows SPOTTER_WAIVE_FEES (or --waive). With --expect
PLATE the output says MINE instead of the plate text. Needs NEBIUS_API_KEY in .env.
With --listen SECONDS the Telegram inbox is read instead, as spotter.run does
while it watches, and each text is answered and timed.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.alerts import TelegramSetupError, make_alerts  # noqa: E402
from spotter.attendant import BASE_URL  # noqa: E402
from spotter.db import Database  # noqa: E402
from spotter.dispute import DISPUTE_MODEL, DisputeAgent, GoodwillRule  # noqa: E402
from spotter.inbox import TelegramInbox  # noqa: E402
from spotter.price import make_price_agent  # noqa: E402
from spotter.privacy import Masker  # noqa: E402
from spotter.reader import normalize  # noqa: E402
from spotter.words import clock_words  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Answer one driver message from the record, as the dispute worker would.")
    parser.add_argument("text", nargs="?", help="the driver's message")
    parser.add_argument("--listen", type=float, metavar="SECONDS", help="read texts sent to the bot for this long instead")
    parser.add_argument("--expect", metavar="PLATE", help="your plate text, so output says MINE instead")
    parser.add_argument("--alerts", choices=("console", "telegram"))
    parser.add_argument("--waive", choices=("on", "off"), help="the goodwill rule switch (SPOTTER_WAIVE_FEES)")
    parser.add_argument("--waive-within-min", type=float, metavar="N")
    parser.add_argument("--model", default=DISPUTE_MODEL)
    args = parser.parse_args()
    if (args.text is None) == (args.listen is None):
        parser.error("give a text, or --listen SECONDS")
    flags = {"alerts": args.alerts, "waive_within_min": args.waive_within_min}
    if args.waive is not None:
        flags["waive_fees"] = args.waive == "on"
    try:
        settings = config.load(flags)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        print("NEBIUS_API_KEY is missing: add it to .env", file=sys.stderr)
        return 2
    masker = Masker(normalize(args.expect) if args.expect else None)
    try:
        alerts = make_alerts(settings.alerts, masker.print)
    except TelegramSetupError as err:
        print(f"Telegram alerts cannot start: {err}", file=sys.stderr)
        return 2
    print(
        f"alerts: {settings.alerts} | model: {args.model} | goodwill rule: waiving {'on' if settings.waive_fees else 'off'},"
        f" within {settings.waive_within_min:g} minutes of the fee beginning | overstay fee {settings.overstay_fee_cents} cents"
    )
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=180, max_retries=0)
    rule = GoodwillRule(settings.waive_fees, settings.waive_within_min, settings.overstay_fee_cents)
    if args.listen is None:
        db = Database()
        agent = DisputeAgent(client, db, alerts, masker, rule, settings.overstay_grace_s, model=args.model)
        try:
            agent.handle(args.text)
        finally:
            alerts.close()
            db.close()
        agent.print_stats()
        return 0

    def answer(text: str, received_at: datetime) -> None:
        """Runs on the inbox thread, with a database connection of its own."""
        got = datetime.now(timezone.utc)
        started = time.perf_counter()
        print(f"INBOX: text received at {clock_words(got)} (sent {clock_words(received_at)}): {text!r}", flush=True)
        own_db = Database()
        try:
            if text.strip().lower() == "/price":
                agent = make_price_agent(client, own_db, alerts, masker, settings)
                how = agent.handle(received_at)
            else:
                agent = DisputeAgent(client, own_db, alerts, masker, rule, settings.overstay_grace_s, model=args.model)
                how = agent.handle(text, received_at)
            reply_at = next(
                (datetime.fromisoformat(a["time"]) for a in own_db.actions()
                 if a["tool"] == "send_alert" and datetime.fromisoformat(a["time"]) >= got.replace(microsecond=0)),
                None,
            )
        finally:
            own_db.close()
        reply = f"{(reply_at - got).total_seconds():.0f}s" if reply_at else "not delivered"
        who = "owner's" if text.strip().lower() == "/price" else "driver's"
        print(f"INBOX: handled ({how}) in {time.perf_counter() - started:.1f}s; the {who} reply went out {reply} after receipt", flush=True)
        agent.print_stats()

    inbox = TelegramInbox.from_env(answer, masker.print)
    inbox.start()
    print(f"listening for texts to the bot for {args.listen:g} seconds (Ctrl-C to stop early)", flush=True)
    deadline = time.monotonic() + args.listen
    try:
        while time.monotonic() < deadline:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    inbox.stop()
    alerts.close()
    print(f"inbox: {inbox.received} text(s) answered, {inbox.ignored} update(s) ignored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
