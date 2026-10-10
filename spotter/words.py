"""Clock times and durations as words, made by code, in the machine's local time zone.

The model repeats what it is given. Give it "6:02 pm" and "16 seconds" and it
cannot turn 16 seconds into 16 minutes, so every time or duration it may
repeat goes through here, and a WordBook remembers what was handed out for an
event so the check can see that no alert states a time code did not supply.
"""

from datetime import datetime, timezone


def clock_words(moment: datetime, now: datetime | None = None) -> str:
    """"6:02 pm", or "Thu 9 Oct, 6:02 pm" when the moment is not today, in local time."""
    local = moment.astimezone()
    time_text = local.strftime("%I:%M %p").lstrip("0").lower()
    today = (now or datetime.now(timezone.utc)).astimezone().date()
    if local.date() == today:
        return time_text
    return f"{local.strftime('%a')} {local.day} {local.strftime('%b')}, {time_text}"


def _unit(count: int, name: str) -> str:
    return f"{count} {name}" + ("" if count == 1 else "s")


def duration_words(seconds: float) -> str:
    """"16 seconds", "1 minute 30 seconds", "1 hour 46 minutes", "2 days 3 hours"."""
    total = max(0, round(seconds))
    if total < 60:
        return _unit(total, "second")
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return _unit(minutes, "minute") + (f" {_unit(secs, 'second')}" if secs else "")
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return _unit(hours, "hour") + (f" {_unit(minutes, 'minute')}" if minutes else "")
    days, hours = divmod(hours, 24)
    return _unit(days, "day") + (f" {_unit(hours, 'hour')}" if hours else "")


class WordBook:
    """Hands out time and duration words and remembers every one given for an event."""

    def __init__(self) -> None:
        self.supplied: list[str] = []

    def clock(self, moment: datetime, now: datetime | None = None) -> str:
        return self.note(clock_words(moment, now))

    def duration(self, seconds: float) -> str:
        return self.note(duration_words(seconds))

    def note(self, words: str) -> str:
        """Record words made elsewhere (the time-left words, for example) and return them."""
        if words not in self.supplied:
            self.supplied.append(words)
        return words
