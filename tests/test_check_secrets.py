"""Tests of scripts/check_secrets.py on text and files in a temp folder, never through git."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_secrets.py"
_spec = importlib.util.spec_from_file_location("check_secrets", SCRIPT)
check_secrets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_secrets)

# Fictional values for the fake .env. Each one is built from pieces, so that no line of this
# file looks like a secret to the check itself when the file is committed.
CAMERA = "camera.test"
SCHEME = "rtsp://"
SOURCE = SCHEME + "user:pw@" + CAMERA + ":554/stream1"  # a placeholder: rule 1 and 2 catch it, rule 4 does not
REAL_LOOKING = SCHEME + "admin:hunter2@" + CAMERA + ":554/stream1"
AREA = "Elm Street, Springfield"
TAVILY = "tvly-" + "test-0123456789abc"
BOT_TOKEN = "123456789:" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
OTHER_BOT_TOKEN = "987654321:" + "A" * 35
NEBIUS = "nebius-" + "fictional-key-0001"
FAKE_ENV = "\n".join([
    "# fictional settings for the tests",
    "export NEBIUS_API_KEY='" + NEBIUS + "'",
    "SPOTTER_SOURCE=" + SOURCE,
    'SPOTTER_LOT_AREA="' + AREA + '"',
    "TAVILY_API_KEY=" + TAVILY,
    "TELEGRAM_BOT_TOKEN=" + BOT_TOKEN + "  # the bot",
    "TELEGRAM_CHAT_ID=4242",
    "SPOTTER_LIGHT=kasa",
    "",
])


def rule_names(text, rules):
    return [rule for _, _, rule in check_secrets.scan_text("t.md", text, rules)]


class TempEnvCase(unittest.TestCase):
    """A temp folder standing in for the repository, with the fake .env in it."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        (self.root / ".env").write_text(FAKE_ENV)
        self.rules, self.notice = check_secrets.build_rules(self.root / ".env")

    def names(self, text, rules=None):
        return rule_names(text, self.rules if rules is None else rules)


class RulesTest(TempEnvCase):
    def test_env_is_parsed_with_export_quotes_and_comments(self):
        env = check_secrets.parse_env(FAKE_ENV)
        self.assertEqual(env["NEBIUS_API_KEY"], NEBIUS)
        self.assertEqual(env["SPOTTER_SOURCE"], SOURCE)
        self.assertEqual(env["SPOTTER_LOT_AREA"], AREA)
        self.assertEqual(env["TELEGRAM_BOT_TOKEN"], BOT_TOKEN)
        self.assertEqual(env["TELEGRAM_CHAT_ID"], "4242")
        self.assertEqual(self.notice, "")

    def test_key_value_is_reported_by_key_name_and_never_shown(self):
        hits = check_secrets.scan_text("notes.md", "first line\nthe key is " + TAVILY + "\n", self.rules)
        self.assertIn(("notes.md", 2, "value of TAVILY_API_KEY"), hits)
        self.assertNotIn(TAVILY, repr(hits))
        self.assertIn("value of NEBIUS_API_KEY", self.names("key " + NEBIUS))
        self.assertIn("value of TELEGRAM_BOT_TOKEN", self.names(BOT_TOKEN))

    def test_values_under_six_characters_are_not_rules(self):
        self.assertEqual(self.names("kasa 4242"), [])

    def test_camera_host(self):
        self.assertEqual(check_secrets.camera_host(SOURCE), CAMERA)
        self.assertEqual(check_secrets.camera_host("samples/demo.mp4"), "")
        self.assertEqual(check_secrets.camera_host(SCHEME + CAMERA + ":554/stream1"), "")
        self.assertIn("camera host", self.names("the camera at " + CAMERA + " answers"))
        self.assertIn("camera host", self.names("CAMERA.TEST:554"))
        self.assertNotIn("camera host", self.names("my" + CAMERA))

    def test_area_word_inside_a_url_slug(self):
        self.assertEqual(check_secrets.area_words(AREA), ["street", "springfield"])
        self.assertEqual(check_secrets.area_words("Springfield, New Jersey"), ["springfield"])  # the state is public
        self.assertIn("lot area word", self.names("https://example.test/springfield-events/tonight"))
        self.assertIn("lot area word", self.names("springfield_events2026"))
        self.assertIn("lot area word", self.names("STREET parking"))
        # "Elm" has three letters, so it is not a rule; a word inside a longer word is not a hit
        self.assertEqual(self.names("elm trees line the streets"), [])

    def test_bot_token_that_is_not_in_env(self):
        self.assertEqual(self.names("token " + OTHER_BOT_TOKEN), ["telegram bot token"])

    def test_patterns(self):
        patterns = check_secrets.PATTERNS
        self.assertEqual(rule_names(REAL_LOOKING, patterns), ["rtsp credentials"])
        for placeholder in (SOURCE, SCHEME + "user:password@" + CAMERA, SCHEME + "***@" + CAMERA + "/x", SCHEME + "USER:PASSWORD@" + CAMERA):
            self.assertEqual(rule_names(placeholder, patterns), [], placeholder)
        self.assertEqual(rule_names(SCHEME + CAMERA + ":554/stream1", patterns), [])
        self.assertEqual(rule_names("sk-" + "abcdefghijklmnop", patterns), ["sk- key"])
        self.assertEqual(rule_names("task-management", patterns), [])
        self.assertEqual(rule_names(TAVILY, patterns), ["tvly- key"])
        self.assertEqual(rule_names("hf_" + "ABCDEFGHIJ0123", patterns), ["hf_ token"])
        self.assertEqual(rule_names("AKIA" + "ABCDEFGHIJKLMNOP", patterns), ["aws access key"])
        self.assertEqual(rule_names("AKIA" + "ABCDEFGHIJKLMNO", patterns), [])

    def test_private_paths(self):
        paths = [
            "data/events.jsonl", "samples/demo.mp4", "samples/car.jpg", "samples_out/car.png",
            ".env", ".env.example", "spotter/run.py", "./data/eval/frame.jpg", "database.py",
        ]
        self.assertEqual(
            [path for path, _, _ in check_secrets.path_hits(paths)],
            ["data/events.jsonl", "samples/car.jpg", "samples_out/car.png", ".env", "./data/eval/frame.jpg"],
        )
        self.assertEqual(check_secrets.path_hits(["data/log.txt"]), [("data/log.txt", 1, "private path")])

    def test_word_ignores_case_spaces_and_hyphens(self):
        rules, _ = check_secrets.build_rules(self.root / ".env", ["MAYR 473"])
        for text in ("plate mayr473 seen", "MAYR 473", "mayr-473 at noon", "m a y r 4 7 3"):
            self.assertIn("forbidden word", self.names(text, rules), text)
        self.assertNotIn("forbidden word", self.names("MAYR 437", rules))
        self.assertNotIn("forbidden word", self.names("plate mayr473 seen"))

    def test_clean_text_passes(self):
        self.assertEqual(self.names("RENT743 arrived at 6:02 pm and left at 8:15 pm; the bill was $5.00."), [])
        self.assertEqual(self.names("SPOTTER_ALERTS=console and SPOTTER_LIGHT=console run with no hardware"), [])

    def test_missing_env_skips_rules_1_to_3_with_a_notice(self):
        rules, notice = check_secrets.build_rules(self.root / "missing" / ".env")
        self.assertIn("no .env", notice)
        self.assertIn("rules 1 to 3", notice)
        self.assertEqual(self.names(CAMERA + " " + TAVILY + " springfield " + NEBIUS, rules), ["tvly- key"])
        self.assertEqual(self.names(OTHER_BOT_TOKEN, rules), ["telegram bot token"])


class DiffTest(unittest.TestCase):
    def test_added_lines_are_scanned_at_their_new_line_numbers(self):
        diff = "\n".join([
            "diff --git a/notes.md b/notes.md",
            "index 1111111..2222222 100644",
            "--- a/notes.md",
            "+++ b/notes.md",
            "@@ -3,0 +4,2 @@ heading",
            "+clean words",
            "+token " + OTHER_BOT_TOKEN,
            "@@ -9 +11 @@",
            "-old " + OTHER_BOT_TOKEN,
            "+new line",
            "diff --git a/samples/car.jpg b/samples/car.jpg",
            "new file mode 100644",
            "index 0000000..3333333",
            "Binary files /dev/null and b/samples/car.jpg differ",
            "diff --git a/gone.txt b/gone.txt",
            "deleted file mode 100644",
            "--- a/gone.txt",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-bye " + OTHER_BOT_TOKEN,
            "",
        ])
        hits = check_secrets.scan_diff(diff, check_secrets.PATTERNS)
        self.assertEqual(hits, [("notes.md", 5, "telegram bot token")])

    def test_empty_diff_has_no_hits(self):
        self.assertEqual(check_secrets.scan_diff("", check_secrets.PATTERNS), [])


class RunTest(TempEnvCase):
    """run() on files given by path; the temp folder is the repository root and git is not used."""

    def run_check(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            code = check_secrets.run(list(argv), repo_root=self.root, out=out)
        return code, out.getvalue()

    def test_hits_name_the_rule_and_never_the_value(self):
        notes = self.root / "notes.md"
        notes.write_text("\n".join([
            "hello", "key " + TAVILY, "cam " + CAMERA, "bot " + OTHER_BOT_TOKEN, "map springfield-events", "",
        ]))
        code, out = self.run_check(str(notes))
        self.assertEqual(code, 1)
        for line in ("notes.md:2 value of TAVILY_API_KEY", "notes.md:2 tvly- key", "notes.md:3 camera host",
                     "notes.md:4 telegram bot token", "notes.md:5 lot area word"):
            self.assertIn(line + "\n", out)
        self.assertRegex(out.strip().splitlines()[-1], r"^\d+ problems?: fix them before committing$")
        for secret in (TAVILY, CAMERA, OTHER_BOT_TOKEN, "springfield", "Springfield"):
            self.assertNotIn(secret, out)

    def test_private_path_is_reported(self):
        log = self.root / "data" / "log.txt"
        log.parent.mkdir()
        log.write_text("clean\n")
        self.assertEqual(self.run_check(str(log)), (1, "data/log.txt:1 private path\n1 problem: fix them before committing\n"))

    def test_clean_file_passes(self):
        clean = self.root / "clean.md"
        clean.write_text("RENT743 stayed 2 hours.\n")
        self.assertEqual(self.run_check(str(clean)), (0, "no secrets found\n"))

    def test_word_from_the_command_line(self):
        log = self.root / "log.txt"
        log.write_text("saw plate mayr 473 at noon\n")
        code, out = self.run_check("--word", "MAYR473", str(log))
        self.assertEqual((code, out), (1, "log.txt:1 forbidden word\n1 problem: fix them before committing\n"))

    def test_folder_means_its_files(self):
        (self.root / "notes").mkdir()
        (self.root / "notes" / "a.md").write_text("fine\n")
        (self.root / "notes" / "b.md").write_text("cam " + CAMERA + "\n")
        self.assertEqual(self.run_check(str(self.root / "notes")), (1, "notes/b.md:1 camera host\n1 problem: fix them before committing\n"))

    def test_missing_env_prints_one_notice(self):
        (self.root / ".env").unlink()
        clean = self.root / "clean.md"
        clean.write_text("nothing here\n")
        code, out = self.run_check(str(clean))
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("notice: no .env"))
        self.assertEqual(lines[1], "no secrets found")

    def test_missing_file_is_an_error(self):
        code, out = self.run_check(str(self.root / "nowhere.md"))
        self.assertEqual((code, out), (2, ""))


if __name__ == "__main__":
    unittest.main()
