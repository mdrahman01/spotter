"""Add a booking that starts now, or empty the database.

    python -m spotter.seed --plate PLATE --driver NAME --hours 12 --rate 5.00
    python -m spotter.seed --reset

With --reset and a plate, the database is emptied first and the booking added
after. The plate is taken from the command line and is not printed.
"""

import argparse
import sys
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from spotter.db import Database, iso, now
from spotter.reader import normalize


def dollars_to_cents(text: str) -> int:
    try:
        return int((Decimal(text) * 100).to_integral_value())
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None


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
        starts_at = now()
        ends_at = starts_at + timedelta(hours=args.hours)
        booking_id = db.add_booking(plate, args.driver, starts_at, ends_at, args.rate)
        print(
            f"booking #{booking_id} added for the plate given on the command line:"
            f" {args.driver}, {args.hours:g}h from {iso(starts_at)} to {iso(ends_at)},"
            f" {args.rate} cents/hour"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
