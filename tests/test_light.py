"""Tests of the Kasa light with a stand-in bulb, including the failure path. No network."""

import unittest

from kasa import Module

from spotter.light import ConsoleLight, KasaLight, LightError, make_light


class FakeLightModule:
    def __init__(self, bulb):
        self.bulb = bulb

    async def set_hsv(self, hue, saturation, value, *, transition=None):
        if self.bulb.broken:
            raise OSError("no route to host")
        self.bulb.calls.append(("hsv", hue, saturation, value, transition))
        self.bulb.on = True


class FakeBulb:
    def __init__(self, broken=False):
        self.broken = broken
        self.calls = []
        self.on = False
        self.modules = {Module.Light: FakeLightModule(self)}

    async def turn_off(self, **kwargs):
        if self.broken:
            raise OSError("no route to host")
        self.calls.append(("off", kwargs.get("transition")))
        self.on = False


class KasaLightTest(unittest.TestCase):
    def make(self, bulb, **kwargs):
        self.connects = 0

        async def connect(host, username, password):
            self.connects += 1
            self.seen = (host, username, password)
            return bulb

        return KasaLight("192.168.1.77", "me@example.com", "secret-pass", connect=connect, say=lambda line: None, **kwargs)

    def test_colours_at_full_saturation_with_no_fade(self):
        bulb = FakeBulb()
        light = self.make(bulb, brightness=80)
        for color in ("green", "amber", "red", "off"):
            light.set(color)
        self.assertEqual(
            bulb.calls,
            [("hsv", 120, 100, 80, 0), ("hsv", 40, 100, 80, 0), ("hsv", 0, 100, 80, 0), ("off", 0)],
        )
        self.assertEqual(light.color, "off")
        self.assertEqual(self.connects, 1)  # one connection for four commands
        self.assertEqual(self.seen, ("192.168.1.77", "me@example.com", "secret-pass"))

    def test_a_failed_command_raises_light_error_and_reconnects_next_time(self):
        bulb = FakeBulb(broken=True)
        light = self.make(bulb)
        with self.assertRaises(LightError) as caught:
            light.set("green")
        self.assertIn("OSError", str(caught.exception))
        self.assertNotIn("secret-pass", str(caught.exception))
        self.assertIsNone(light.color)
        bulb.broken = False
        light.set("green")
        self.assertEqual(light.color, "green")
        self.assertEqual(self.connects, 2)

    def test_unknown_colour(self):
        with self.assertRaises(LightError):
            self.make(FakeBulb()).set("blue")

    def test_make_light(self):
        self.assertIsInstance(make_light("console", None, 100, lambda line: None), ConsoleLight)
        self.assertIsInstance(make_light("kasa", "192.168.1.77", 100, lambda line: None), KasaLight)
        with self.assertRaises(LightError):
            make_light("kasa", None, 100)
        with self.assertRaises(LightError):
            make_light("laser", None, 100)


if __name__ == "__main__":
    unittest.main()
