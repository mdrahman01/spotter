"""The signal light: a console stand-in today, the Kasa bulb once it arrives."""

from collections.abc import Callable
from typing import Protocol

COLORS = ("green", "amber", "red", "off")


class Light(Protocol):
    color: str | None  # what the light shows now, None until first set

    def set(self, color: str) -> None:
        """Show green, amber, red or off."""


class ConsoleLight:
    """Prints "LIGHT -> GREEN" instead of switching a bulb, and remembers every change."""

    def __init__(self, say: Callable[[str], None] = print) -> None:
        self.say = say
        self.history: list[str] = []
        self.color: str | None = None

    def set(self, color: str) -> None:
        self.history.append(color)
        self.color = color
        self.say(f"LIGHT -> {color.upper()}")


class KasaLight:
    """Where the TP-Link Kasa bulb goes (pip install python-kasa).

    Plan: connect to the bulb by host, then set_hsv(120, 100, 100) for green,
    set_hsv(40, 100, 100) for amber, set_hsv(0, 100, 100) for red and turn_off()
    for off. Until the bulb is here
    this raises, so nothing can pretend to have switched it.
    """

    def __init__(self, host: str) -> None:
        self.host = host
        self.color: str | None = None

    def set(self, color: str) -> None:
        raise NotImplementedError("the Kasa bulb is not wired up yet")
