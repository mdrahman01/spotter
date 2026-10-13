"""Hand one driver message to the dispute worker without Telegram.

    .venv/bin/python scripts/dispute.py "I left at 6:50, why the fee?" [--expect PLATE] [--alerts console|telegram]
                                        [--waive on|off] [--waive-within-min 10]

The message counts as coming from the driver of the stay that ended most
recently. The reply and the owner's note go where SPOTTER_ALERTS (or --alerts)
says; the goodwill rule follows SPOTTER_WAIVE_FEES (or --waive). With --expect
PLATE the output says MINE instead of the plate text. Needs NEBIUS_API_KEY in .env.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter import config  # noqa: E402
from spotter.alerts import TelegramSetupError, make_alerts  # noqa: E402
from spotter.attendant import BASE_URL  # noqa: E402
from spotter.db import Database  # noqa: E402
from spotter.dispute import DISPUTE_MODEL, DisputeAgent, GoodwillRule  # noqa: E402
from spotter.privacy import Masker  # noqa: E402
from spotter.reader import normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Answer one driver message from the record, as the dispute worker would.")
    parser.add_argument("text", help="the driver's message")
    parser.add_argument("--expect", metavar="PLATE", help="your plate text, so output says MINE instead")
    parser.add_argument("--alerts", choices=("console", "telegram"))
    parser.add_argument("--waive", choices=("on", "off"), help="the goodwill rule switch (SPOTTER_WAIVE_FEES)")
    parser.add_argument("--waive-within-min", type=float, metavar="N")
    parser.add_argument("--model", default=DISPUTE_MODEL)
    args = parser.parse_args()
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
    db = Database()
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=180, max_retries=0)
    rule = GoodwillRule(settings.waive_fees, settings.waive_within_min, settings.overstay_fee_cents)
    agent = DisputeAgent(client, db, alerts, masker, rule, settings.overstay_grace_s, model=args.model)
    try:
        agent.handle(args.text)
    finally:
        alerts.close()
        db.close()
    agent.print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
