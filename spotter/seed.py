"""Add a booking that starts now, or empty the database.

    python -m spotter.seed --plate PLATE --driver NAME --hours 12 --rate 5.00
    python -m spotter.seed --plate PLATE --driver NAME --rate 5.00 \
        --starts-at 2026-10-07T03:00:00+00:00 --ends-at 2026-10-07T03:01:12+00:00
    python -m spotter.seed --reset

With --reset and a plate, the database is emptied first and the booking added
after. The plate is taken from the command line and is not printed.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from spotter.db import Database, iso, now
from spotter.reader import normalize


def dollars_to_cents(text: str) -> int:
    try:
        return int((Decimal(text) * 100).to_integral_value())
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None


def parse_time(text: str) -> datetime:
    """An ISO 8601 time; one without a zone is taken as UTC."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spotter.seed",
        description="Add a booking that starts now, or empty the database.",
    )
    parser.add_argument("--plate", help="the booked plate")
    parser.add_argument("--driver", help="the driver's name")
    parser.add_argument("--hours", type=float, default=12, help="booking length (default 12)")
    parser.add_argument(
        "--rate", type=dollars_to_cents, default="5.00", metavar="DOLLARS",
        help="dollars per hour (default 5.00)",
    )
    parser.add_argument(
        "--starts-at", type=parse_time, metavar="ISO_TIME", help="booking start (default: now)"
    )
    parser.add_argument(
        "--ends-at", type=parse_time, metavar="ISO_TIME", help="booking end (default: start plus --hours)"
    )
    parser.add_argument("--reset", action="store_true", help="empty the database first")
    args = parser.parse_args()
    if not args.reset and not args.plate:
        parser.error("nothing to do: pass --plate ... or --reset")
    if args.plate and not args.driver:
        parser.error("--plate needs --driver")
    if args.hours <= 0:
        parser.error("--hours must be above 0")

    db = Database()
    if args.reset:
        db.reset()
        print("database emptied")
    if args.plate:
        plate = normalize(args.plate)
        if not plate:
            parser.error("--plate needs at least one letter or digit")
        starts_at = args.starts_at or now()
        ends_at = args.ends_at or starts_at + timedelta(hours=args.hours)
        if ends_at <= starts_at:
            parser.error("--ends-at must be after the start")
        booking_id = db.add_booking(plate, args.driver, starts_at, ends_at, args.rate)
        print(
            f"booking #{booking_id} added for the plate given on the command line:"
            f" {args.driver}, from {iso(starts_at)} to {iso(ends_at)}, {args.rate} cents/hour"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
