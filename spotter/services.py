"""Which outside services were used, when, and how fast: for the dashboard's footer.

Every model call, scene look and web search notes itself here, in memory, so
the page can show each service's last latency and whether it was used in the
last ten minutes. Nothing is stored; a restart starts blank.
"""

import threading
import time
from dataclasses import dataclass

LIGHTNING = "Nemotron 3.5 Lightning on Nebius Token Factory"
ULTRA = "Nemotron 3 Ultra on Nebius Token Factory"
COSMOS = "Cosmos Reason 2 on Nebius AI Cloud"
TAVILY = "Tavily search"
SERVICES = (LIGHTNING, ULTRA, COSMOS, TAVILY)
RECENT_S = 10 * 60


@dataclass
class Use:
    last_at: float | None = None  # time.time()
    last_latency_s: float | None = None
    calls: int = 0


_uses: dict[str, Use] = {name: Use() for name in SERVICES}
_lock = threading.Lock()


def note(service: str, latency_s: float) -> None:
    """Record one use of a service; unknown names are kept too."""
    with _lock:
        use = _uses.setdefault(service, Use())
        use.last_at = time.time()
        use.last_latency_s = latency_s
        use.calls += 1


def model_service(model: str) -> str:
    return ULTRA if "ultra" in model.lower() else LIGHTNING


def snapshot(now: float | None = None) -> list[dict]:
    """One entry per service, in footer order."""
    now = time.time() if now is None else now
    with _lock:
        return [
            {
                "name": name,
                "last_latency_s": round(use.last_latency_s, 2) if use.last_latency_s is not None else None,
                "last_used_s_ago": round(now - use.last_at) if use.last_at is not None else None,
                "recent": use.last_at is not None and now - use.last_at <= RECENT_S,
                "calls": use.calls,
            }
            for name, use in _uses.items()
        ]


def reset() -> None:
    with _lock:
        for use in _uses.values():
            use.last_at, use.last_latency_s, use.calls = None, None, 0
