"""Keeps plate text out of printed output when --expect is given."""

import re


class Masker:
    """Replaces plate text with MINE or OTHER in everything that is printed.

    With no expected plate, text passes through unchanged and plates print as
    their text. Otherwise the expected plate prints as MINE and the plate of the
    event being handled prints as MINE or OTHER, also when the model writes it
    with spaces or hyphens in between.
    """

    def __init__(self, expected: str | None) -> None:
        self.expected = expected
        self._rules: list[tuple[re.Pattern[str], str]] = []
        if expected:
            self._rules.append((_pattern(expected), "MINE"))

    def label(self, plate: str) -> str:
        """How this plate is shown: its text, or MINE/OTHER with --expect."""
        if self.expected is None:
            return plate
        return "MINE" if plate == self.expected else "OTHER"

    def for_event(self, plate: str) -> None:
        """Also mask the plate of the event now being handled."""
        if self.expected is not None and plate != self.expected:
            self._rules = self._rules[:1] + [(_pattern(plate), "OTHER")]

    def mask(self, text: str) -> str:
        for pattern, label in self._rules:
            text = pattern.sub(label, text)
        return text

    def print(self, line: str) -> None:
        print(self.mask(line), flush=True)


def _pattern(plate: str) -> re.Pattern[str]:
    return re.compile(r"[\s\-]*".join(re.escape(ch) for ch in plate), re.IGNORECASE)
