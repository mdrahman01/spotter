"""Tests of the pricing agent with no network: scripted searches and a scripted model."""

import json
import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from spotter import config
from spotter.alerts import ConsoleAlerts
from spotter.db import Database
from spotter.price import AFTER_REMINDER, BY_CODE, CLEAN, MAX_SEARCHES, PRICE_MODEL, PRICE_PROMPT, PriceAgent, PriceToolbox, TavilySearch, make_price_agent, price_until
from spotter.privacy import Masker
from spotter.seed import tonight_rate
from spotter.words import WordBook

NOW = datetime(2026, 10, 13, 23, 5, tzinfo=timezone.utc)  # 7:05 pm in New York
RESULTS = [
    {"title": "Big Arena: Concert tonight", "url": "https://example.com/arena", "content": "Doors 7 pm at Big Arena, 2 miles away.", "published_date": "2026-10-13"},
    {"title": "Street fair this weekend", "url": "https://example.com/fair", "content": "Saturday and Sunday.", "published_date": None},
]


def quiet(line: str) -> None:
    pass


class FakeTavily(TavilySearch):
    def __init__(self, results=RESULTS, fail=False):
        self.queries = []
        self.results = results
        self.fail = fail
        super().__init__("tvly-secret", post=self._post, get=lambda url, headers: {"key": {"usage": 7}})

    def _post(self, url, body, headers):
        self.queries.append(body["query"])
        assert headers["Authorization"] == "Bearer tvly-secret"
        assert (body["search_depth"], body["max_results"]) == ("basic", 6)
        if self.fail:
            raise OSError("no route to api.tavily.com with tvly-secret")
        return {"results": self.results}


class FakeModel:
    api_key = "fake-key"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self.create)))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.script.pop(0) if self.script else "Done."
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            message = {"role": "assistant", "content": reply}
        else:
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(reply)]}
        body = {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 9}}}
        return SimpleNamespace(text=json.dumps(body))


SEARCH = ("search_events", {"query": "events tonight near the lot"})
PEAK = ("set_demand", {"level": "peak", "event": "Concert at Big Arena, doors 7 pm", "reason": "arena-scale concert tonight 2 miles away", "source_url": "https://example.com/arena"})
BUSY = ("set_demand", {"level": "busy", "event": "Concert at Big Arena, doors 7 pm", "reason": "will fill street parking", "source_url": "https://example.com/arena"})
BASE = ("keep_base", {"reason": "nothing on tonight within the radius"})


def owner(message):
    return ("send_alert", {"to": "owner", "message": message})


class Scene:
    def __init__(self, script, results=RESULTS, fail=False, env=None):
        self.db = Database(":memory:")
        self.settings = config.load({}, env=env or {"SPOTTER_LOT_AREA": "Elm Street, Springfield"})
        self.alerts = ConsoleAlerts(say=quiet)
        self.tavily = FakeTavily(results, fail)
        self.model = FakeModel(script)
        masker = Masker(None)
        masker.print = quiet
        masker.hide(self.settings.lot_area, "<lot area>")
        self.masker = masker
        self.agent = PriceAgent(self.model, self.db, self.alerts, masker, self.settings, self.tavily)

    def rows(self, tool=None):
        return [a for a in self.db.actions() if tool is None or a["tool"] == tool]


class DecisionTest(unittest.TestCase):
    def test_a_peak_decision_is_accepted_and_the_owner_told_with_the_link(self):
        scene = Scene([[SEARCH], [PEAK], [owner("Tonight's price is $12.00 an hour (peak) until 6:00 am: Concert at Big Arena, doors 7 pm. Source: https://example.com/arena")], "Done."])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        price = scene.db.tonight_price(NOW)
        self.assertEqual((price["level"], price["rate_cents_per_hour"], price["source_url"]), ("peak", 1200, "https://example.com/arena"))
        self.assertEqual(price["valid_until"], price_until(NOW).isoformat(timespec="seconds"))
        decision = json.loads(scene.rows("set_demand")[0]["result"])
        self.assertEqual(decision["price"], "$12.00 an hour")  # the price comes from code, as words
        words = json.loads(scene.rows("words")[0]["arguments"])["supplied"]
        self.assertIn("$12.00 an hour", words)
        payload = json.loads(scene.model.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["radius_miles"], 5)
        self.assertRegex(payload["today"], r"^\w{3} \w{3} \d{1,2}, 2026$")
        self.assertRegex(payload["local_time"], r"\d{1,2}:\d{2} (am|pm)")
        self.assertEqual(scene.agent.stats.searches, 1)
        self.assertEqual(scene.agent.stats.reasoning_tokens, [9, 9, 9, 9])

    def test_a_busy_decision_sets_the_busy_price(self):
        scene = Scene([[SEARCH], [BUSY], [owner("Tonight's price is $8.00 an hour (busy) until 6:00 am: Concert at Big Arena. https://example.com/arena")], "Done."])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        self.assertEqual(scene.db.tonight_price(NOW)["rate_cents_per_hour"], 800)

    def test_a_link_that_was_not_in_the_results_is_refused(self):
        bad = ("set_demand", {"level": "peak", "event": "x", "reason": "y", "source_url": "https://example.com/made-up"})
        scene = Scene([[SEARCH], [bad], [BASE], [owner("Tonight's price stays at $5.00 an hour (base): no verified event.")], "Done."])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        refused = json.loads(scene.rows("set_demand")[0]["result"])
        self.assertIn("refused", refused["error"])
        self.assertEqual(scene.db.tonight_price(NOW)["level"], "base")
        self.assertEqual(scene.db.tonight_price(NOW)["rate_cents_per_hour"], 500)

    def test_a_decision_before_any_search_with_an_unknown_link_and_a_second_decision_are_refused(self):
        scene = Scene([[PEAK], [SEARCH], [PEAK], [BASE], [owner("Tonight's price is $12.00 an hour (peak): concert. https://example.com/arena")], "Done."])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        first, second = (json.loads(r["result"]) for r in scene.rows("set_demand"))
        self.assertIn("refused", first["error"])  # no search yet, so no known url
        self.assertEqual(second["level"], "peak")
        self.assertIn("already made", json.loads(scene.rows("keep_base")[0]["result"])["error"])
        self.assertIn("send_alert the owner", json.loads(scene.rows("keep_base")[0]["result"])["error"])

    def test_a_fourth_search_is_refused(self):
        scene = Scene([[SEARCH], [SEARCH], [SEARCH], [SEARCH], [BASE], [owner("Tonight's price stays at $5.00 an hour (base).")], "Done."])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        results = [json.loads(r["result"]) for r in scene.rows("search_events")]
        self.assertEqual([("error" in r) for r in results], [False, False, False, True])
        self.assertIn("limit", results[3]["error"])
        self.assertEqual(len(scene.tavily.queries), MAX_SEARCHES)
        self.assertEqual(results[0]["searches_left"], 2)

    def test_no_decision_means_code_keeps_the_base_price_and_says_so(self):
        scene = Scene([[SEARCH], "I am not sure.", "Still not sure."])
        self.assertEqual(scene.agent.handle(NOW), BY_CODE)
        told = json.loads(scene.rows("reminder")[0]["result"])["told"]
        self.assertIn("exactly one decision", told)
        price = scene.db.tonight_price(NOW)
        self.assertEqual((price["level"], price["rate_cents_per_hour"]), ("base", 500))
        by_code = json.loads(scene.rows("keep_base")[0]["arguments"])
        self.assertEqual(by_code["by"], "code")
        message = scene.alerts.sent[-1].message
        self.assertIn("$5.00 an hour", message)
        self.assertIn("stays", message)
        self.assertEqual(scene.agent.stats.code_items, 2)

    def test_an_owner_message_without_the_link_is_reminded(self):
        scene = Scene([[SEARCH], [PEAK], [owner("Tonight's price is $12.00 an hour (peak): a concert.")], "Done.",
                       [owner("Tonight's price is $12.00 an hour (peak): Concert at Big Arena. Source: https://example.com/arena")], "Added the link."])
        self.assertEqual(scene.agent.handle(NOW), AFTER_REMINDER)
        self.assertIn("source link", json.loads(scene.rows("reminder")[0]["result"])["told"])

    def test_a_failed_search_is_an_error_not_a_crash_and_hides_the_key(self):
        scene = Scene([[SEARCH], [BASE], [owner("Tonight's price stays at $5.00 an hour (base): the search failed.")], "Done."], fail=True)
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        result = json.loads(scene.rows("search_events")[0]["result"])
        self.assertIn("search failed", result["error"])
        self.assertNotIn("tvly-secret", result["error"])

    def test_the_lot_area_is_hidden_in_output(self):
        printed = []
        scene = Scene([[SEARCH], [BASE], [owner("Tonight's price stays at $5.00 an hour (base).")], "Done."])
        scene.masker.print = printed.append
        scene.agent.handle(NOW)
        self.assertTrue(printed)
        self.assertNotIn("Elm Street", scene.masker.mask("events near Elm Street, Springfield tonight"))


class ExpiryAndSeedTest(unittest.TestCase):
    def test_tonights_price_lasts_until_6_am_local(self):
        until = price_until(NOW)
        local = until.astimezone()
        self.assertEqual((local.hour, local.minute), (6, 0))
        self.assertGreater(until, NOW)
        self.assertLessEqual(until - NOW, timedelta(hours=24))
        self.assertEqual(price_until(until), until + timedelta(days=1))  # at 6 am sharp, the next day's 6 am

    def test_seed_uses_tonights_price_until_it_expires(self):
        db = Database(":memory:")
        until = price_until(NOW)
        self.assertEqual(tonight_rate(db, None, NOW)[1], "base price")
        self.assertEqual(tonight_rate(db, 700, NOW), (700, "given"))
        db.set_price("busy", 800, NOW, until, "Concert", "busy night", "https://example.com/arena")
        self.assertEqual(tonight_rate(db, None, NOW), (800, "tonight's price, busy"))
        self.assertEqual(tonight_rate(db, None, until - timedelta(minutes=1))[0], 800)
        self.assertEqual(tonight_rate(db, None, until)[1], "base price")  # expired at 6 am
        self.assertIsNone(db.tonight_price(NOW - timedelta(hours=1)))  # not before it was set

    def test_rates_come_from_env_as_dollars(self):
        settings = config.load({}, env={"SPOTTER_RATE_BASE": "5.00", "SPOTTER_RATE_BUSY": "$8.50", "SPOTTER_RATE_PEAK": "12"})
        self.assertEqual((settings.rate_base, settings.rate_busy, settings.rate_peak), (500, 850, 1200))
        with self.assertRaises(ValueError):
            config.load({}, env={"SPOTTER_RATE_PEAK": "twelve"})
        with self.assertRaises(ValueError):
            config.load({}, env={"SPOTTER_PRICE_CHECK_AT": "4pm"})
        self.assertEqual(config.load({}, env={"SPOTTER_PRICE_CHECK_AT": "16:00"}).price_check_at, "16:00")


class OwnerMessageRulesTest(unittest.TestCase):
    def test_the_message_needs_a_decision_the_price_and_the_link_and_goes_once(self):
        scene = Scene([
            [SEARCH],
            [owner("Tonight is busy.")],  # before any decision: refused
            [PEAK],
            [owner("Tonight is peak because of the concert.")],  # no price: refused
            [owner("Tonight's price is $12.00 an hour (peak): the concert at Big Arena.")],  # no link: refused
            [owner("Tonight's price is $12.00 an hour (peak): Concert at Big Arena, doors 7 pm. https://example.com/arena")],
            [owner("Tonight's price is $12.00 an hour (peak): Concert at Big Arena, doors 7 pm. https://example.com/arena")],  # again: refused
            "Done.",
        ])
        self.assertEqual(scene.agent.handle(NOW), CLEAN)
        results = [json.loads(r["result"]) for r in scene.rows("send_alert")]
        self.assertIn("decide first", results[0]["error"])
        self.assertIn("$12.00 an hour", results[1]["error"])
        self.assertIn("source link", results[2]["error"])
        self.assertTrue(results[3]["delivered"])
        self.assertIn("already told", results[4]["error"])
        self.assertEqual(len(scene.alerts.sent), 1)  # the owner got exactly one message


class ToolboxTest(unittest.TestCase):
    def test_set_demand_needs_filled_fields(self):
        db = Database(":memory:")
        toolbox = PriceToolbox(db, ConsoleAlerts(say=quiet), FakeTavily(), config.load({}, env={}), WordBook(), NOW)
        toolbox.search_events("events")
        self.assertIn("filled in", toolbox.set_demand("busy", "", "why", "https://example.com/arena")["error"])
        self.assertIn("busy or peak", toolbox.set_demand("base", "x", "why", "https://example.com/arena")["error"])
        self.assertIn("filled in", toolbox.keep_base(" ")["error"])
        self.assertEqual(toolbox.call("set_demand", "not json")["error"], "arguments are not valid JSON")
        self.assertIn("to must be owner", toolbox.send_alert("hi", to="driver")["error"])
        self.assertIn("one argument", toolbox.call("send_alert", json.dumps({"level": "base", "price": "$5.00 an hour"}))["error"])
        toolbox.keep_base("nothing on")
        self.assertTrue(toolbox.call("send_alert", json.dumps({"message": "The price stays at $5.00 an hour."}))["delivered"])


class GuideTest(unittest.TestCase):
    def test_the_guide_carries_the_owners_policy_right_after_rule_2(self):
        guide = " ".join(PRICE_PROMPT.split())  # on one line, so wrapping cannot split a phrase
        self.assertIn("an event counts only if it draws more cars than the streets near the lot can hold", guide)
        self.assertIn("A stadium or arena event is peak", guide)
        self.assertIn("A concert venue, festival, parade, college game or large fair is busy", guide)
        self.assertIn("A show, sale, class or meeting at a hall, lodge, school or shop is not an event for pricing", guide)
        self.assertEqual(re.findall(r"(?m)^(\d+)\. ", PRICE_PROMPT), ["1", "2", "3", "4", "5", "6"])
        self.assertTrue(guide.index("2. peak:") < guide.index("3. The owner's policy") < guide.index("4. End the check"))
        # the policy narrows rule 2 and leaves the old rules standing
        self.assertIn("never widens it", guide)
        self.assertIn("whenever you are unsure", guide)
        self.assertIn("means the base price", guide)
        self.assertIn("exactly one decision", guide)


class ModelChoiceTest(unittest.TestCase):
    ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
    LIGHTNING = "nvidia/Nemotron-3_5-Lightning"

    def test_the_careful_judge_is_the_default_price_model(self):
        self.assertEqual(PRICE_MODEL, self.ULTRA)
        self.assertEqual(config.Settings().price_model, self.ULTRA)
        self.assertEqual(config.load({}, env={}).price_model, self.ULTRA)
        self.assertEqual(Scene([]).agent.model, self.ULTRA)  # PriceAgent's own default
        help_text = config.Settings.__dataclass_fields__["price_model"].metadata["help"]
        self.assertIn("careful judge", help_text)
        self.assertIn("SPOTTER_PRICE_MODEL", help_text)

    def test_spotter_price_model_still_switches_the_model(self):
        self.assertEqual(config.load({}, env={"SPOTTER_PRICE_MODEL": self.LIGHTNING}).price_model, self.LIGHTNING)
        self.assertEqual(config.load({"price_model": "other/model"}, env={"SPOTTER_PRICE_MODEL": self.LIGHTNING}).price_model, "other/model")  # --model wins
        for env_model, expected in ((None, self.ULTRA), (self.LIGHTNING, self.LIGHTNING)):
            env = {"SPOTTER_LOT_AREA": "Elm Street, Springfield"}
            if env_model:
                env["SPOTTER_PRICE_MODEL"] = env_model
            masker = Masker(None)
            masker.print = quiet
            with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
                agent = make_price_agent(FakeModel([]), Database(":memory:"), ConsoleAlerts(say=quiet), masker, config.load({}, env=env))
            self.assertEqual(agent.model, expected)


if __name__ == "__main__":
    unittest.main()
