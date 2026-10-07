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
    amount_cents INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_session_per_plate
    ON sessions (plate) WHERE status = 'open';
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

    def reset(self) -> None:
        """Empty every table."""
        with self.conn:
            for table in ("action_log", "sessions", "bookings"):
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

    def session(self, session_id: int) -> dict | None:
        return self._one("SELECT * FROM sessions WHERE id = ?", session_id)

    def sessions(self) -> list[dict]:
        return self._all("SELECT * FROM sessions ORDER BY id")

    def start_session(self, plate: str, booking_id: int, arrived_at: datetime) -> int:
        """Open a stay. Raises sqlite3.IntegrityError if the plate already has one open."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sessions (plate, booking_id, arrived_at, status)"
                " VALUES (?, ?, ?, 'open')",
                (plate, booking_id, iso(arrived_at)),
            )
        return cursor.lastrowid

    def close_session(self, session_id: int, left_at: datetime) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET left_at = ?, status = 'closed' WHERE id = ?",
                (iso(left_at), session_id),
            )

    def set_amount(self, session_id: int, amount_cents: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET amount_cents = ? WHERE id = ?", (amount_cents, session_id)
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
