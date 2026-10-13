"""Keeps plate text out of printed output when --expect is given."""

import re

COMMON_WORDS = {"street", "road", "avenue", "drive", "lane", "north", "south", "east", "west", "near", "area", "between", "corner", "block", "city", "town"}
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


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
        self._hidden: list[tuple[re.Pattern[str], str]] = []
        if expected:
            self._rules.append((_pattern(expected), "MINE"))

    def hide(self, text: str | None, label: str, words_too: bool = True) -> None:
        """Also replace `text` (the lot area, say) with `label` in everything printed; with
        words_too every word of it of four letters or more is replaced as well, since a model
        or a search result will rephrase "Elm Street, Springfield" as "Springfield" or "ELM ST"."""
        if not text or not text.strip():
            return
        self._hidden.append((re.compile(re.escape(text.strip()), re.IGNORECASE), label))
        if words_too:
            for word in sorted({w for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text)}, key=len, reverse=True):
                if word.lower() in COMMON_WORDS:
                    continue
                boundary = r"\b" if len(word) < 5 else ""  # a longer word is hidden inside slugs and domains too
                self._hidden.append((re.compile(boundary + re.escape(word) + boundary, re.IGNORECASE), label))
            for name, code in US_STATES.items():
                if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
                    self._hidden.append((re.compile(r"\b" + code + r"\b"), label))

    def label(self, plate: str) -> str:
        """How this plate is shown: its text, or MINE/OTHER with --expect; VEHICLE when there is no plate."""
        if not plate:
            return "VEHICLE"
        if self.expected is None:
            return plate
        return "MINE" if plate == self.expected else "OTHER"

    def for_event(self, plate: str) -> None:
        """Also mask the plate of the event now being handled."""
        if plate and self.expected is not None and plate != self.expected:
            self._rules = self._rules[:1] + [(_pattern(plate), "OTHER")]

    def mask(self, text: str) -> str:
        for pattern, label in self._rules + self._hidden:
            text = pattern.sub(label, text)
        return text

    def print(self, line: str) -> None:
        print(self.mask(line), flush=True)


def _pattern(plate: str) -> re.Pattern[str]:
    return re.compile(r"[\s\-]*".join(re.escape(ch) for ch in plate), re.IGNORECASE)
