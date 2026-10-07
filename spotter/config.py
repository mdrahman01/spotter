"""Settings for the watcher, with defaults that env vars or CLI flags override.

Precedence, lowest first: the defaults below, SPOTTER_* environment variables
(e.g. SPOTTER_LEAVE_AFTER_S=60), then command-line flags (e.g. --leave-after-s 60).
The zone is written x0,y0,x1,y1, e.g. SPOTTER_ZONE=0.35,0.27,0.69,0.69.
"""

import argparse
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields

ENV_PREFIX = "SPOTTER_"

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


def parse_zone(text: str) -> Zone:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"zone needs four numbers x0,y0,x1,y1, got {text!r}")
    x0, y0, x1, y1 = (float(part) for part in parts)
    return x0, y0, x1, y1


PARSERS: dict[str, Callable[[str], object]] = {
    "source": str,
    "zone": parse_zone,
    "process_fps": float,
    "min_det_conf": float,
    "arrive_reads": int,
    "arrive_window_s": float,
    "leave_after_s": float,
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
        parser.add_argument(
            "--" + setting.name.replace("_", "-"),
            type=checked,
            default=None,
            help=f"{setting.metadata['help']} "
            f"(default {default}; env {env_name(setting.name)})",
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
