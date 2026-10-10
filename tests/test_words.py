"""Tests of the words code makes for times and durations."""

import re
import unittest
from datetime import datetime, timedelta, timezone

from spotter.words import WordBook, clock_words, duration_words


class DurationWordsTest(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(duration_words(0), "0 seconds")
        self.assertEqual(duration_words(1), "1 second")
        self.assertEqual(duration_words(16.4), "16 seconds")
        self.assertEqual(duration_words(59.4), "59 seconds")

    def test_minutes_and_seconds(self):
        self.assertEqual(duration_words(60), "1 minute")
        self.assertEqual(duration_words(90), "1 minute 30 seconds")
        self.assertEqual(duration_words(16 * 60), "16 minutes")
        self.assertEqual(duration_words(59 * 60 + 59), "59 minutes 59 seconds")

    def test_hours_and_days(self):
        self.assertEqual(duration_words(3600), "1 hour")
        self.assertEqual(duration_words(3600 + 46 * 60 + 20), "1 hour 46 minutes")
        self.assertEqual(duration_words(2 * 3600), "2 hours")
        self.assertEqual(duration_words(26 * 3600 + 5), "1 day 2 hours")
        self.assertEqual(duration_words(48 * 3600), "2 days")

    def test_never_negative(self):
        self.assertEqual(duration_words(-5), "0 seconds")


class ClockWordsTest(unittest.TestCase):
    def test_today_is_time_only_in_local_time(self):
        now = datetime.now(timezone.utc)
        words = clock_words(now, now)
        self.assertRegex(words, r"^\d{1,2}:\d{2} (am|pm)$")
        self.assertEqual(words, now.astimezone().strftime("%I:%M %p").lstrip("0").lower())

    def test_another_day_names_the_day(self):
        now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        moment = now - timedelta(days=3)
        words = clock_words(moment, now)
        local = moment.astimezone()
        self.assertTrue(words.startswith(f"{local.strftime('%a')} {local.day} {local.strftime('%b')}, "))
        self.assertRegex(words, r", \d{1,2}:\d{2} (am|pm)$")

    def test_no_leading_zero(self):
        local_morning = datetime(2026, 10, 9, 6, 2).astimezone()
        self.assertTrue(clock_words(local_morning, local_morning).startswith("6:02 "))


class WordBookTest(unittest.TestCase):
    def test_remembers_each_phrase_once(self):
        book = WordBook()
        now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        a = book.clock(now, now)
        b = book.duration(16)
        book.duration(16)
        c = book.note("less than a minute")
        self.assertEqual(book.supplied, [a, b, c])
        self.assertEqual(re.sub(r"\d", "N", b), "NN seconds")


if __name__ == "__main__":
    unittest.main()
