"""SQLite storage at data/spotter.db: bookings, stays (sessions) and the action log."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "spotter.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY,
    plate TEXT NOT NULL,
    driver TEXT NOT NULL,
    starts_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    rate_cents_per_hour INTEGER NOT NULL CHECK (rate_cents_per_hour >= 0)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    plate TEXT NOT NULL,
    booking_id INTEGER NOT NULL REFERENCES bookings (id),
    arrived_at TEXT NOT NULL,
    left_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    amount_cents INTEGER,
    ending_soon_at TEXT,
    overstayed_at TEXT,
    fee_began_at TEXT,
    left_snapshot TEXT,
    arrival_snapshot TEXT,
    overstay_snapshot TEXT,
    fee_waived_at TEXT,
    waiver_reason TEXT,
    waived_cents INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_session_per_plate
    ON sessions (plate) WHERE status = 'open';
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY,
    set_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('base', 'busy', 'peak')),
    rate_cents_per_hour INTEGER NOT NULL CHECK (rate_cents_per_hour >= 0),
    event TEXT,
    reason TEXT,
    source_url TEXT
);
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    event TEXT NOT NULL,
    tool TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT NOT NULL
);
"""


def iso(moment: datetime) -> str:
    """UTC ISO 8601 to the second: the one format stored, so strings sort in time order."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        # Databases created before the timer events, the snapshots and the fee
        # waiver lack these columns.
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(sessions)")}
        for column, kind in (
            ("ending_soon_at", "TEXT"), ("overstayed_at", "TEXT"), ("fee_began_at", "TEXT"), ("left_snapshot", "TEXT"),
            ("arrival_snapshot", "TEXT"), ("overstay_snapshot", "TEXT"),
            ("fee_waived_at", "TEXT"), ("waiver_reason", "TEXT"), ("waived_cents", "INTEGER"),
        ):
            if column not in columns:
                self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {kind}")
        self.conn.commit()

    def reset(self) -> None:
        """Empty every table."""
        with self.conn:
            for table in ("action_log", "sessions", "bookings", "prices"):
                self.conn.execute(f"DELETE FROM {table}")

    def _one(self, sql: str, *params) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, *params) -> list[dict]:
        return [dict(row) for row in self.conn.execute(sql, params)]

    # bookings

    def add_booking(
        self, plate: str, driver: str, starts_at: datetime, ends_at: datetime, rate_cents_per_hour: int
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO bookings (plate, driver, starts_at, ends_at, rate_cents_per_hour)"
                " VALUES (?, ?, ?, ?, ?)",
                (plate, driver, iso(starts_at), iso(ends_at), rate_cents_per_hour),
            )
        return cursor.lastrowid

    def booking(self, booking_id: int) -> dict | None:
        return self._one("SELECT * FROM bookings WHERE id = ?", booking_id)

    def booking_at(self, plate: str, moment: datetime) -> dict | None:
        """The booking for the plate whose window covers `moment`, if any."""
        return self._one(
            "SELECT * FROM bookings WHERE plate = ? AND starts_at <= ? AND ? < ends_at"
            " ORDER BY starts_at LIMIT 1",
            plate, iso(moment), iso(moment),
        )

    def latest_booking(self, plate: str) -> dict | None:
        return self._one(
            "SELECT * FROM bookings WHERE plate = ? ORDER BY ends_at DESC LIMIT 1", plate
        )

    # sessions

    def open_session(self, plate: str) -> dict | None:
        return self._one("SELECT * FROM sessions WHERE plate = ? AND status = 'open'", plate)

    def open_sessions(self) -> list[dict]:
        return self._all("SELECT * FROM sessions WHERE status = 'open' ORDER BY id")

    def session(self, session_id: int) -> dict | None:
        return self._one("SELECT * FROM sessions WHERE id = ?", session_id)

    def sessions(self) -> list[dict]:
        return self._all("SELECT * FROM sessions ORDER BY id")

    def start_session(self, plate: str, booking_id: int, arrived_at: datetime, snapshot: str | None = None) -> int:
        """Open a stay, keeping the arrival frame. Raises sqlite3.IntegrityError if the plate already has one open."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sessions (plate, booking_id, arrived_at, status, arrival_snapshot)"
                " VALUES (?, ?, ?, 'open', ?)",
                (plate, booking_id, iso(arrived_at), snapshot),
            )
        return cursor.lastrowid

    def latest_closed_session(self) -> dict | None:
        """The stay that ended most recently, if any: the one a driver's message is about."""
        return self._one("SELECT * FROM sessions WHERE status = 'closed' ORDER BY left_at DESC, id DESC LIMIT 1")

    def close_session(self, session_id: int, left_at: datetime, snapshot: str | None = None) -> None:
        """Close a stay at `left_at`, keeping the path of the frame that last showed the car."""
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET left_at = ?, status = 'closed', left_snapshot = ? WHERE id = ?",
                (iso(left_at), snapshot, session_id),
            )

    def close(self) -> None:
        self.conn.close()

    def mark_ending_soon(self, session_id: int, at: datetime) -> None:
        """Record that the ending-soon warning was raised, so it is raised once."""
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET ending_soon_at = ? WHERE id = ?", (iso(at), session_id)
            )

    def mark_overstayed(self, session_id: int, at: datetime, fee_began: datetime | None = None) -> None:
        """Record that the stay overstayed its booking at the sighting `at`, and when the fee began
        (the booking's end plus the grace period); compute_bill then adds the fee."""
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET overstayed_at = ?, fee_began_at = ? WHERE id = ?",
                (iso(at), iso(fee_began) if fee_began else None, session_id),
            )

    def set_overstay_snapshot(self, session_id: int, snapshot: str) -> None:
        """Keep the frame of the overstay sighting: the photo a dispute is answered with."""
        with self.conn:
            self.conn.execute("UPDATE sessions SET overstay_snapshot = ? WHERE id = ?", (snapshot, session_id))

    def waive_fee(self, session_id: int, at: datetime, reason: str, cents: int) -> None:
        """Take the overstay fee off a closed stay's bill and record why: the only way money changes after billing."""
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET fee_waived_at = ?, waiver_reason = ?, waived_cents = ?,"
                " amount_cents = amount_cents - ? WHERE id = ?",
                (iso(at), reason, cents, cents, session_id),
            )

    def fee_waived_before(self, plate: str) -> bool:
        """Has this plate ever had a fee waived? The goodwill rule allows it once."""
        return self._one("SELECT id FROM sessions WHERE plate = ? AND fee_waived_at IS NOT NULL LIMIT 1", plate) is not None

    def set_amount(self, session_id: int, amount_cents: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET amount_cents = ? WHERE id = ?", (amount_cents, session_id)
            )

    # tonight's price

    def set_price(self, level: str, rate_cents: int, set_at: datetime, valid_until: datetime,
                  event: str | None = None, reason: str | None = None, source_url: str | None = None) -> int:
        """Record tonight's price: the level the pricing agent chose and the price the owner's table gives it."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO prices (set_at, valid_until, level, rate_cents_per_hour, event, reason, source_url)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (iso(set_at), iso(valid_until), level, rate_cents, event, reason, source_url),
            )
        return cursor.lastrowid

    def tonight_price(self, moment: datetime) -> dict | None:
        """The price in force at `moment`: the latest record whose valid_until is still ahead."""
        return self._one(
            "SELECT * FROM prices WHERE set_at <= ? AND valid_until > ? ORDER BY id DESC LIMIT 1", iso(moment), iso(moment)
        )

    # action log

    def log_action(self, event: dict, tool: str, arguments, result) -> None:
        """Record one thing the attendant did, at the current time."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO action_log (time, event, tool, arguments, result) VALUES (?, ?, ?, ?, ?)",
                (
                    iso(now()),
                    json.dumps(event),
                    tool,
                    arguments if isinstance(arguments, str) else json.dumps(arguments),
                    json.dumps(result),
                ),
            )

    def actions(self) -> list[dict]:
        return self._all("SELECT * FROM action_log ORDER BY id")
