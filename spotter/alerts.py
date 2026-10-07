"""Alerts to the owner or a driver: a console stand-in today, phones later."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Alert:
    to: str  # "owner" or "driver"
    name: str | None  # the driver's name, if known
    message: str
    snapshot: str  # path of the event's snapshot, attached by the code


class Alerts(Protocol):
    def send(self, alert: Alert) -> None:
        """Deliver one alert."""


class ConsoleAlerts:
    """Prints "ALERT to owner: ..." instead of messaging anyone, and keeps what was sent."""

    def __init__(self, say: Callable[[str], None] = print) -> None:
        self.say = say
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)
        who = alert.to if alert.name is None else f"{alert.to} ({alert.name})"
        self.say(f"ALERT to {who}: {alert.message} [snapshot: {alert.snapshot}]")


class PhoneAlerts:
    """Where phone alerts go (e.g. a push service or SMS gateway).

    Plan: the owner's number comes from settings, a driver's from the booking,
    and the snapshot is attached as an image. Until then this raises, so nothing
    can pretend to have sent anything.
    """

    def __init__(self, owner_number: str) -> None:
        self.owner_number = owner_number

    def send(self, alert: Alert) -> None:
        raise NotImplementedError("phone alerts are not wired up yet")
