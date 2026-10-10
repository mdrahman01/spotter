"""Find Kasa bulbs on the network, or run the signal colours on one.

    .venv/bin/python scripts/light_test.py                 # no host: list bulbs found on the network
    .venv/bin/python scripts/light_test.py --host 192.168.1.77   # or SPOTTER_LIGHT_HOST from .env

With a host it shows green, amber, red and off for 2 s each and prints how long
each command took. KASA_USERNAME and KASA_PASSWORD come from .env if the bulb
needs them; neither is printed.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from spotter.light import KasaLight, LightError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


async def discover(username, password) -> None:
    from kasa import Discover

    print("looking for Kasa devices for 5 s...")
    found = await Discover.discover(discovery_timeout=5, username=username, password=password)
    if not found:
        print("none found: is this machine on the same Wi-Fi as the bulb?")
        return
    for host, device in sorted(found.items()):
        try:
            await device.update()
            name, model = device.alias, device.model
        except Exception as err:  # some devices need the account to answer
            name, model = f"(no answer: {type(err).__name__})", "?"
        print(f"  {host:<16} {name!s:<24} {model}")
    print("put the bulb's host in .env as SPOTTER_LIGHT_HOST")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Find Kasa bulbs, or run the colours on one.")
    parser.add_argument("--host", default=os.environ.get("SPOTTER_LIGHT_HOST") or None)
    parser.add_argument("--brightness", type=int, default=int(os.environ.get("SPOTTER_LIGHT_BRIGHTNESS") or 100))
    args = parser.parse_args()
    username = os.environ.get("KASA_USERNAME") or None
    password = os.environ.get("KASA_PASSWORD") or None

    if not args.host:
        asyncio.run(discover(username, password))
        return 0
    light = KasaLight(args.host, username, password, args.brightness)
    for color in ("green", "amber", "red", "off"):
        started = time.perf_counter()
        try:
            light.set(color)
        except LightError as err:
            print(f"{color}: FAILED after {time.perf_counter() - started:.2f}s: {err}")
            return 1
        print(f"{color}: {time.perf_counter() - started:.2f}s")
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
