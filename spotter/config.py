"""Settings for the watcher and the attendant, with defaults that env vars or CLI flags override.

Precedence, lowest first: the defaults below, SPOTTER_* environment variables
(e.g. SPOTTER_LEAVE_AFTER_S=60), then command-line flags (e.g. --leave-after-s 60).
The zone is written x0,y0,x1,y1, e.g. SPOTTER_ZONE=0.35,0.27,0.69,0.69.
"""

import argparse
import os
import re
from decimal import Decimal, InvalidOperation
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields

ENV_PREFIX = "SPOTTER_"
ALERT_KINDS = ("console", "telegram")
LIGHT_KINDS = ("console", "kasa")

Zone = tuple[float, float, float, float]


@dataclass(frozen=True)
class Settings:
    source: str | None = field(
        default=None, metadata={"help": "video file path or rtsp:// URL"}
    )
    zone: Zone = field(
        default=(0.35, 0.27, 0.69, 0.69),
        metadata={"help": "watch zone x0,y0,x1,y1 as fractions of the frame"},
    )
    process_fps: float = field(
        default=5, metadata={"help": "frames analysed per second"}
    )
    min_det_conf: float = field(
        default=0.5, metadata={"help": "ignore plate detections below this confidence"}
    )
    arrive_reads: int = field(
        default=5, metadata={"help": "a plate arrives after this many reads of the same text"}
    )
    arrive_window_s: float = field(
        default=3, metadata={"help": "... within this many seconds"}
    )
    leave_after_s: float = field(
        default=4,
        metadata={
            "help": "a plate leaves after this many seconds unread "
            "(4 suits the short test video; live use sets 60)"
        },
    )
    alerts: str = field(
        default="console",
        metadata={"help": "where alerts go: console (prints) or telegram (your own chat)"},
    )
    min_charge_cents: int = field(
        default=100, metadata={"help": "the smallest bill for a stay, in cents"}
    )
    ending_soon_s: float = field(
        default=600,
        metadata={"help": "warn the driver when the booking ends within this many seconds"},
    )
    overstay_grace_s: float = field(
        default=300,
        metadata={"help": "a stay counts as overstayed this many seconds after its booking ends"},
    )
    overstay_fee_cents: int = field(
        default=500, metadata={"help": "flat fee added to an overstayed stay, in cents"}
    )
    light: str = field(
        default="console",
        metadata={"help": "the signal light: console (prints) or kasa (a TP-Link bulb)"},
    )
    light_host: str | None = field(
        default=None, metadata={"help": "IP address or host name of the Kasa bulb"}
    )
    light_brightness: int = field(
        default=100, metadata={"help": "brightness of the bulb's colours, 1 to 100"}
    )
    waive_fees: bool = field(
        default=False,
        metadata={"help": "goodwill rule: may a disputed overstay fee be waived (on/off)"},
    )
    waive_within_min: float = field(
        default=10,
        metadata={"help": "goodwill rule: the car's last read must be within this many minutes of the fee beginning"},
    )
    scene: str = field(
        default="off",
        metadata={"help": "the second pair of eyes: off, cosmos (COSMOS_URL/COSMOS_TOKEN from .env) or script:<file>"},
    )
    spot_description: str = field(
        default="",
        metadata={"help": "the parking spot as Cosmos is told it, e.g. the paved driveway in the foreground"},
    )
    scene_check_after_s: float = field(
        default=8, metadata={"help": "a present plate unread this long gets a look"}
    )
    scene_relook_every_s: float = field(
        default=15, metadata={"help": "look again this often while the scene still sees the car"}
    )
    scene_idle_every_s: float = field(
        default=15, metadata={"help": "look this often while no car is present"}
    )
    scene_unreadable_after_s: float = field(
        default=20, metadata={"help": "a vehicle seen this long with no plate read is UNREADABLE_VEHICLE"}
    )
    scene_timeout_s: float = field(
        default=15, metadata={"help": "a look slower than this counts as scene unavailable"}
    )
    rate_base: int = field(default=500, metadata={"help": "the owner's base hourly price, in dollars (e.g. 5.00)"})
    rate_busy: int = field(default=800, metadata={"help": "the hourly price on a busy night, in dollars"})
    rate_peak: int = field(default=1200, metadata={"help": "the hourly price on a peak night, in dollars"})
    lot_area: str = field(default="", metadata={"help": "where the lot is, for the event search (never printed)"})
    event_radius_miles: float = field(default=5, metadata={"help": "events within this many miles count"})
    nearby_venues: str = field(default="", metadata={"help": "venues inside the radius, comma-separated, that code searches for today before the judge's first turn (at most four are used)"})
    price_check_at: str = field(default="", metadata={"help": "local time HH:MM of the daily price check (empty: none)"})
    price_model: str = field(default="nvidia/Nemotron-3-Ultra-550b-a55b", metadata={"help": "the model that decides tonight's price: Nemotron 3 Ultra, the careful judge; SPOTTER_PRICE_MODEL still switches it (Lightning followed the tool rules better, but a money decision wants the careful judge)"})
    dashboard: bool = field(default=False, metadata={"help": "serve the owner's page at http://localhost:<dashboard_port> (on/off)"})
    dashboard_port: int = field(default=8000, metadata={"help": "the dashboard's port"})
    dashboard_hold_s: float = field(default=0, metadata={"help": "keep serving the dashboard this long after a replay ends"})
    lot_name: str = field(default="My driveway", metadata={"help": "the lot's name in the dashboard header"})

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.zone
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(
                "zone must be x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and "
                f"0 <= y0 < y1 <= 1, got {self.zone}"
            )
        if self.process_fps <= 0:
            raise ValueError("process_fps must be above 0")
        if not 0 <= self.min_det_conf <= 1:
            raise ValueError("min_det_conf must be between 0 and 1")
        if self.arrive_reads < 1:
            raise ValueError("arrive_reads must be at least 1")
        if self.arrive_window_s <= 0 or self.leave_after_s <= 0:
            raise ValueError("arrive_window_s and leave_after_s must be above 0")
        if self.alerts not in ALERT_KINDS:
            raise ValueError(f"alerts must be one of {', '.join(ALERT_KINDS)}, got {self.alerts!r}")
        if self.min_charge_cents < 0:
            raise ValueError("min_charge_cents must not be negative")
        if self.ending_soon_s < 0 or self.overstay_grace_s < 0 or self.overstay_fee_cents < 0:
            raise ValueError("ending_soon_s, overstay_grace_s and overstay_fee_cents must not be negative")
        if self.light not in LIGHT_KINDS:
            raise ValueError(f"light must be one of {', '.join(LIGHT_KINDS)}, got {self.light!r}")
        if not 1 <= self.light_brightness <= 100:
            raise ValueError("light_brightness must be between 1 and 100")
        if self.waive_within_min < 0:
            raise ValueError("waive_within_min must not be negative")
        if self.scene not in ("off", "cosmos") and not self.scene.startswith("script:"):
            raise ValueError(f"scene must be off, cosmos or script:<file>, got {self.scene!r}")
        if min(self.scene_check_after_s, self.scene_relook_every_s, self.scene_idle_every_s,
               self.scene_unreadable_after_s, self.scene_timeout_s) <= 0:
            raise ValueError("the scene_* seconds must be above 0")
        if min(self.rate_base, self.rate_busy, self.rate_peak) < 0:
            raise ValueError("the rates must not be negative")
        if self.event_radius_miles <= 0:
            raise ValueError("event_radius_miles must be above 0")
        if self.price_check_at and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.price_check_at):
            raise ValueError(f"price_check_at must be HH:MM, got {self.price_check_at!r}")
        if not 0 <= self.dashboard_port <= 65535:
            raise ValueError("dashboard_port must be between 0 (any free port) and 65535")
        if self.dashboard_hold_s < 0:
            raise ValueError("dashboard_hold_s must not be negative")
        if not self.price_model.strip():
            raise ValueError("price_model must not be empty")


def parse_zone(text: str) -> Zone:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"zone needs four numbers x0,y0,x1,y1, got {text!r}")
    x0, y0, x1, y1 = (float(part) for part in parts)
    return x0, y0, x1, y1


def parse_dollars(text: str) -> int:
    """An amount in dollars ("5.00") as whole cents."""
    try:
        return int((Decimal(text.strip().lstrip("$")) * 100).to_integral_value())
    except InvalidOperation:
        raise ValueError(f"not an amount in dollars: {text!r}") from None


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in ("on", "true", "yes", "1"):
        return True
    if value in ("off", "false", "no", "0"):
        return False
    raise ValueError(f"expected on or off, got {text!r}")


PARSERS: dict[str, Callable[[str], object]] = {
    "source": str,
    "zone": parse_zone,
    "process_fps": float,
    "min_det_conf": float,
    "arrive_reads": int,
    "arrive_window_s": float,
    "leave_after_s": float,
    "alerts": str.lower,
    "min_charge_cents": int,
    "ending_soon_s": float,
    "overstay_grace_s": float,
    "overstay_fee_cents": int,
    "light": str.lower,
    "light_host": str,
    "light_brightness": int,
    "waive_fees": parse_bool,
    "waive_within_min": float,
    "scene": str.lower,
    "spot_description": str,
    "scene_check_after_s": float,
    "scene_relook_every_s": float,
    "scene_idle_every_s": float,
    "scene_unreadable_after_s": float,
    "scene_timeout_s": float,
    "rate_base": parse_dollars,
    "rate_busy": parse_dollars,
    "rate_peak": parse_dollars,
    "lot_area": str,
    "event_radius_miles": float,
    "nearby_venues": str,
    "price_check_at": str,
    "price_model": str,
    "dashboard": parse_bool,
    "dashboard_port": int,
    "dashboard_hold_s": float,
    "lot_name": str,
}


def env_name(name: str) -> str:
    return ENV_PREFIX + name.upper()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one flag per setting, e.g. --leave-after-s. A flag left out falls back
    to its env var, then to the default."""
    for setting in fields(Settings):
        parse = PARSERS[setting.name]

        def checked(text: str, parse=parse) -> object:
            try:
                return parse(text)
            except ValueError as err:
                raise argparse.ArgumentTypeError(str(err)) from None

        default = setting.default
        if setting.name == "zone":
            default = ",".join(str(v) for v in default)
        extra = {"nargs": "?", "const": True} if parse is parse_bool else {}  # --dashboard alone means on
        parser.add_argument(
            "--" + setting.name.replace("_", "-"),
            type=checked,
            default=None,
            help=f"{setting.metadata['help']} "
            f"(default {default}; env {env_name(setting.name)})",
            **extra,
        )


def load(
    flags: Mapping[str, object] | None = None, env: Mapping[str, str] | None = None
) -> Settings:
    """Settings from the defaults, then SPOTTER_* env vars, then the flags that were given.

    Raises ValueError for a malformed env var or an invalid combination.
    """
    env = os.environ if env is None else env
    flags = flags or {}
    values = {}
    for setting in fields(Settings):
        raw = env.get(env_name(setting.name), "").strip()
        if raw:
            try:
                values[setting.name] = PARSERS[setting.name](raw)
            except ValueError as err:
                raise ValueError(f"{env_name(setting.name)}: {err}") from None
        if flags.get(setting.name) is not None:
            values[setting.name] = flags[setting.name]
    return Settings(**values)
