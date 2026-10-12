"""The signal light: a console stand-in, or a TP-Link Kasa bulb through python-kasa.

SPOTTER_LIGHT=console (the default) prints "LIGHT -> GREEN". SPOTTER_LIGHT=kasa
drives the bulb at SPOTTER_LIGHT_HOST: green, amber, red and off at full
saturation with no fade, at the brightness SPOTTER_LIGHT_BRIGHTNESS. Newer bulbs
want the Kasa account: KASA_USERNAME and KASA_PASSWORD are read from .env only
then, and never printed. A command the bulb does not answer raises LightError;
the tool turns that into an error result for the model, and the finish check
takes its normal path.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Protocol

COLORS = ("green", "amber", "red", "off")
HUES = {"green": 120, "amber": 40, "red": 0}  # degrees; saturation is always 100
KASA_TIMEOUT_S = 10.0


class LightError(Exception):
    """The light did not do what it was told."""


class Light(Protocol):
    color: str | None  # what the light shows now, None until first set

    def set(self, color: str) -> None:
        """Show green, amber, red or off. Raises LightError if the light does not respond."""


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


async def connect_kasa(host: str, username: str | None, password: str | None):
    """The bulb at `host`, updated and ready; its Light module does the colours.

    Discovery is tried first with the account, if any. A KL125 sometimes
    announces the newer KLAP protocol and then fails the login handshake, while
    it answers the legacy local protocol without any account; so when the first
    way fails, the bulb is opened the legacy way before giving up.
    """
    from kasa import Device, DeviceConfig, DeviceConnectionParameters, DeviceEncryptionType, DeviceFamily, Discover

    first_error = None
    try:
        device = await Discover.discover_single(host, username=username, password=password, timeout=int(KASA_TIMEOUT_S))
        if device is None:
            raise LightError(f"no Kasa device answered at {host}")
        await device.update()
        return device
    except Exception as err:  # wrong protocol guess, or an account the bulb does not know
        first_error = err
    legacy = DeviceConfig(
        host=host,
        timeout=int(KASA_TIMEOUT_S),
        connection_type=DeviceConnectionParameters(DeviceFamily.IotSmartBulb, DeviceEncryptionType.Xor),
    )
    try:
        device = await Device.connect(config=legacy)
        await device.update()
        return device
    except Exception as err:
        raise LightError(f"{type(first_error).__name__} then {type(err).__name__}: {err}") from None


class KasaLight:
    """A Kasa bulb (KL125) through python-kasa: full saturation, no fade, set brightness.

    Connects on first use and again after any failure. `connect` is replaced by
    a stand-in in tests.
    """

    def __init__(
        self,
        host: str,
        username: str | None = None,
        password: str | None = None,
        brightness: int = 100,
        connect: Callable[[str, str | None, str | None], Awaitable] = connect_kasa,
        timeout_s: float = KASA_TIMEOUT_S,
        say: Callable[[str], None] = print,
    ) -> None:
        self.host = host
        self._username = username
        self._password = password
        self.brightness = brightness
        self._connect = connect
        self.timeout_s = timeout_s
        self.say = say
        self.color: str | None = None
        self._device = None
        # One event loop for the life of the light: the bulb's connection belongs to
        # the loop that opened it, so a fresh loop per command would break it.
        self._loop: asyncio.AbstractEventLoop | None = None

    def _run(self, coroutine):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(asyncio.wait_for(coroutine, self.timeout_s))

    def set(self, color: str) -> None:
        if color not in COLORS:
            raise LightError(f"no such colour {color!r}")
        try:
            self._run(self._set(color))
        except Exception as err:  # the bulb is unreachable, slow, or refused
            self._device = None  # reconnect next time
            raise LightError(f"{type(err).__name__}: {err}".replace(str(self._password), "***")) from None
        self.color = color
        self.say(f"LIGHT -> {color.upper()} (Kasa bulb at {self.host})")

    def close(self) -> None:
        """Drop the connection and the event loop."""
        if self._device is not None and self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(asyncio.wait_for(self._device.disconnect(), 5))
            except Exception:
                pass
        self._device = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()

    async def _set(self, color: str) -> None:
        from kasa import Module

        if self._device is None:
            self._device = await self._connect(self.host, self._username, self._password)
        if color == "off":
            await self._device.turn_off(transition=0)
            return
        light = self._device.modules[Module.Light]
        await light.set_hsv(HUES[color], 100, self.brightness, transition=0)


def make_light(kind: str, host: str | None, brightness: int, say: Callable[[str], None] = print) -> Light:
    """The light named by the SPOTTER_LIGHT setting."""
    if kind == "console":
        return ConsoleLight(say)
    if kind == "kasa":
        if not host:
            raise LightError("SPOTTER_LIGHT=kasa needs SPOTTER_LIGHT_HOST")
        return KasaLight(
            host,
            os.environ.get("KASA_USERNAME") or None,
            os.environ.get("KASA_PASSWORD") or None,
            brightness,
            say=say,
        )
    raise LightError(f"unknown light {kind!r}")
