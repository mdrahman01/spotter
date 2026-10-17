#!/usr/bin/env python3
"""Stop a commit that would publish a secret, the camera's address, the lot's area or a private file.

    python3 scripts/check_secrets.py                  # the staged diff, added lines only (what the hook runs)
    python3 scripts/check_secrets.py --all            # every tracked file (git ls-files)
    python3 scripts/check_secrets.py FEEDBACK.md      # the files or folders given
    python3 scripts/check_secrets.py --word MAYR473   # an extra forbidden word, such as the owner's plate

Standard library only, because the pre-commit hook runs it under the system python3. Every hit is
printed as "path:line rule" and the text that matched is never printed. The exit code is 1 when
there is a hit, with a last line "N problems: fix them before committing", and 0 with the line
"no secrets found" when there is none. scripts/install_hooks.sh installs the hook.

The rules:

1. Every value in .env of six or more characters, reported by its key, as in "value of TAVILY_API_KEY".
   The file is read here, one KEY=VALUE per line, with "export " and quotes stripped and comments ignored.
2. The camera host from SPOTTER_SOURCE, the part between the at sign and the next colon or slash,
   reported as "camera host".
3. Every word of SPOTTER_LOT_AREA of four or more letters, in any case, also where hyphens, slashes or
   digits join it to other words as in a web address, reported as "lot area word".
4. Six patterns, each reported by name: an rtsp URL with a password in it ("rtsp credentials";
   placeholders such as user:password or *** do not count), sk- keys
   ("sk- key"), tvly- keys ("tvly- key"), hf_ tokens ("hf_ token"), Telegram bot tokens ("telegram bot
   token") and AWS access keys ("aws access key").
5. Paths under samples/ (except samples/demo.mp4), samples_out/ or data/, and the file .env itself,
   reported as "private path".
6. Words given with --word, in any case and ignoring spaces and hyphens, reported as "forbidden word".
   The owner gives the real plate this way, on the command line only, so it is written in no file.

Without a .env, rules 1 to 3 are skipped and one notice line says so.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_VALUE_CHARS = 6
MIN_WORD_LETTERS = 4
PRIVATE_DIRS = ("samples/", "samples_out/", "data/")
ALLOWED_PRIVATE_PATHS = ("samples/demo.mp4",)
LETTER = r"[^\W\d_]"  # one letter in any alphabet
TOKEN_START = r"(?<![A-Za-z0-9])"  # the match must start a token, so "task-management" is no sk- key

Rule = Tuple[str, "re.Pattern[str]"]  # a name and the regex that finds it
Hit = Tuple[str, int, str]  # path, line number, rule name

# Rule 4. The rtsp pattern is written in two pieces so that this line does not match itself.
PATTERNS: List[Rule] = [
    # Placeholder credentials as the docs and tests write them (user:password, user:pw, ***, USER:PASS)
    # are not a secret; a real user name and password in front of the at sign are.
    ("rtsp credentials", re.compile(r"rtsp:/" r"/(?!(?:\*+|user(?:name)?:(?:password|pass|pw|secret|\*+)|[A-Z_]+:[A-Z_]+)@)[^\s/@]*@")),
    ("sk- key", re.compile(TOKEN_START + r"sk-[A-Za-z0-9]{10,}")),
    ("tvly- key", re.compile(TOKEN_START + r"tvly-[A-Za-z0-9-]{10,}")),
    ("hf_ token", re.compile(TOKEN_START + r"hf_[A-Za-z0-9]{10,}")),
    ("telegram bot token", re.compile(r"(?<![0-9])[0-9]{8,10}:[A-Za-z0-9_-]{35}")),
    ("aws access key", re.compile(TOKEN_START + r"AKIA[0-9A-Z]{16}")),
]


class CheckError(Exception):
    """Something kept the check from running, such as git failing or a path that does not exist."""


# Rules 1 to 3 come from .env.


def unquote(value: str) -> str:
    """The value of a KEY=VALUE line without its quotes, or without a trailing comment when unquoted."""
    value = value.strip()
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        return value[1:end] if end > 0 else value[1:]
    cut = value.find(" #")
    return value[:cut].strip() if cut >= 0 else value


def parse_env(text: str) -> Dict[str, str]:
    """The KEY=VALUE pairs of a .env file. Comments and blank lines are skipped, "export " is dropped."""
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = unquote(value)
    return values


def camera_host(source: str) -> str:
    """The host in a URL with a password, the part between the at sign and the next colon or slash."""
    _, sep, rest = source.partition("://")
    authority = rest.split("/", 1)[0]
    if not sep or "@" not in authority:
        return ""
    return authority.rsplit("@", 1)[1].split(":", 1)[0].strip()


# The words of the US state names: the state in "Town, New Jersey" is public, and a state table in
# spotter/privacy.py names them all, so they are not the part of the lot's area that is kept private.
STATE_WORDS = frozenset(
    "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii "
    "idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan "
    "minnesota mississippi missouri montana nebraska nevada hampshire jersey mexico york north south "
    "carolina dakota ohio oklahoma oregon pennsylvania rhode island tennessee texas utah vermont "
    "virginia washington west wisconsin wyoming".split()
)


def area_words(area: str) -> List[str]:
    """The distinct words of four or more letters in the lot's area, lower case, in order,
    leaving out the words of US state names."""
    words: List[str] = []
    for word in re.findall(LETTER + "{%d,}" % MIN_WORD_LETTERS, area):
        if word.lower() not in words and word.lower() not in STATE_WORDS:
            words.append(word.lower())
    return words


def env_rules(env: Dict[str, str]) -> List[Rule]:
    """Rules 1 to 3 from the parsed .env: its long values, the camera host and the lot area words."""
    rules: List[Rule] = []
    for key, value in env.items():
        if len(value) >= MIN_VALUE_CHARS:
            rules.append(("value of " + key, re.compile(re.escape(value))))
    host = camera_host(env.get("SPOTTER_SOURCE", ""))
    if host:
        rules.append(("camera host", re.compile(TOKEN_START + re.escape(host) + r"(?![A-Za-z0-9])", re.I)))
    for word in area_words(env.get("SPOTTER_LOT_AREA", "")):
        regex = re.compile("(?<!%s)%s(?!%s)" % (LETTER, re.escape(word), LETTER), re.I)
        rules.append(("lot area word", regex))
    return rules


def word_rules(words: Sequence[str]) -> List[Rule]:
    """Rule 6: each --word, in any case, with any spaces or hyphens between its characters."""
    rules: List[Rule] = []
    for word in words:
        letters = re.sub(r"[\s-]", "", word)
        if letters:
            pattern = r"[\s-]*".join(re.escape(ch) for ch in letters)
            rules.append(("forbidden word", re.compile(pattern, re.I)))
    return rules


def build_rules(env_path: Path, words: Sequence[str] = ()) -> Tuple[List[Rule], str]:
    """All rules for a run, and a notice line when .env is missing (empty otherwise)."""
    notice = ""
    rules: List[Rule] = []
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        notice = (
            "notice: no .env at %s, so rules 1 to 3 were skipped "
            "(its values, the camera host and the lot area words)" % env_path
        )
    else:
        rules.extend(env_rules(parse_env(text)))
    rules.extend(PATTERNS)
    rules.extend(word_rules(words))
    return rules, notice


# Scanning: text, files and the staged diff. A hit names the path, the line and the rule, never the text.


def scan_lines(path: str, lines: Iterable[Tuple[int, str]], rules: Sequence[Rule]) -> List[Hit]:
    """Every (path, line number, rule name) where a rule matches one of the numbered lines."""
    hits: List[Hit] = []
    for number, text in lines:
        for name, regex in rules:
            if regex.search(text):
                hits.append((path, number, name))
    return hits


def scan_text(path: str, text: str, rules: Sequence[Rule]) -> List[Hit]:
    """Scan a whole text, numbering its lines from 1."""
    return scan_lines(path, enumerate(text.splitlines(), 1), rules)


def is_private(path: str) -> bool:
    """Rule 5: is this repository path one that must never be committed?"""
    clean = posixpath.normpath(path.replace(os.sep, "/"))
    if posixpath.basename(clean) == ".env":
        return True
    if clean in ALLOWED_PRIVATE_PATHS:
        return False
    return clean.startswith(PRIVATE_DIRS)


def path_hits(paths: Iterable[str]) -> List[Hit]:
    """Rule 5 over a list of repository paths. A private path is reported at line 1."""
    return [(path, 1, "private path") for path in paths if is_private(path)]


def scan_files(files: Iterable[Tuple[str, Path]], rules: Sequence[Rule]) -> List[Hit]:
    """Scan files given as (path to show, path to read). Binary files (a NUL byte early on) are skipped."""
    hits: List[Hit] = []
    for shown, actual in files:
        try:
            data = actual.read_bytes()
        except OSError as err:
            raise CheckError("cannot read %s: %s" % (shown, err.strerror or err)) from None
        if b"\0" in data[:8000]:
            continue
        hits.extend(scan_text(shown, data.decode("utf-8", "replace"), rules))
    return hits


def diff_path(header: str) -> Optional[str]:
    """The path in a "+++ b/path" header, or None for /dev/null (a deleted file)."""
    path = header[4:].strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path == "/dev/null":
        return None
    return path[2:] if path.startswith("b/") else path


def scan_diff(diff: str, rules: Sequence[Rule]) -> List[Hit]:
    """Scan the added lines of a unified diff, each reported at its line number in the new file."""
    hits: List[Hit] = []
    path: Optional[str] = None
    in_header = False
    line = 0
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            in_header, path = True, None
        elif in_header and raw.startswith("+++ "):
            path = diff_path(raw)
        elif raw.startswith("@@"):
            in_header = False
            found = re.match(r"@@ -\S+ \+(\d+)", raw)
            line = int(found.group(1)) if found else 0
        elif in_header:
            continue
        elif raw.startswith("+"):
            if path is not None:
                hits.extend(scan_lines(path, [(line, raw[1:])], rules))
            line += 1
        elif raw.startswith(" "):
            line += 1  # a context line; there are none with -U0
    return hits


# Finding what to scan.


def git(args: Sequence[str], repo_root: Path) -> str:
    """The output of a git command run in the repository, or a CheckError."""
    try:
        done = subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True, check=True)
    except FileNotFoundError:
        raise CheckError("git is not installed or not on PATH") from None
    except subprocess.CalledProcessError as err:
        message = err.stderr.decode("utf-8", "replace").strip() or "exit code %d" % err.returncode
        raise CheckError("git %s failed: %s" % (" ".join(args[:2]), message)) from None
    return done.stdout.decode("utf-8", "replace")


def split_paths(output: str) -> List[str]:
    return [path for path in output.split("\0") if path]


def tracked_files(repo_root: Path) -> List[str]:
    return split_paths(git(["ls-files", "-z"], repo_root))


def staged_paths(repo_root: Path) -> List[str]:
    """The paths of staged files that will exist after the commit (deletions are not a leak)."""
    return split_paths(git(["diff", "--cached", "--name-only", "-z", "--diff-filter=d"], repo_root))


def staged_diff(repo_root: Path) -> str:
    """The staged diff with no context lines, plain prefixes and no external diff tool."""
    return git(
        ["diff", "--cached", "-U0", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/"],
        repo_root,
    )


def walk(folder: Path) -> Iterator[Path]:
    """Every file under a folder, in order, skipping hidden folders such as .git and .venv."""
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            yield Path(dirpath) / name


def listed_files(given: Sequence[str], repo_root: Path) -> List[Tuple[str, Path]]:
    """The files named on the command line as (path to show, path to read); a folder means its files."""
    root = repo_root.resolve()
    pairs: List[Tuple[str, Path]] = []
    for item in given:
        actual = Path(item).expanduser().resolve()
        if not actual.exists():
            raise CheckError("no such file: %s" % item)
        for file in [actual] if actual.is_file() else walk(actual):
            try:
                shown = file.relative_to(root).as_posix()
            except ValueError:  # outside the repository: show it as given
                shown = item if file == actual else str(file)
            pairs.append((shown, file))
    return pairs


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    summary, _, rest = (__doc__ or "").strip().partition("\n")
    parser = argparse.ArgumentParser(
        prog="check_secrets.py", description=summary, epilog=rest.strip("\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", metavar="PATH", help="files or folders to scan instead of the staged diff")
    parser.add_argument("--all", action="store_true", help="scan every tracked file (git ls-files)")
    parser.add_argument(
        "--word", action="append", default=[], metavar="WORD",
        help="an extra forbidden word, in any case and ignoring spaces and hyphens; repeatable",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None, repo_root: Path = REPO_ROOT, out: TextIO = sys.stdout) -> int:
    """Scan, print one line per hit and a summary line, and return the exit code (0, 1 or 2)."""
    args = parse_args(argv)
    rules, notice = build_rules(repo_root / ".env", args.word)
    if notice:
        print(notice, file=out)
    try:
        if args.all:
            paths = tracked_files(repo_root)
            hits = path_hits(paths) + scan_files([(path, repo_root / path) for path in paths], rules)
        elif args.paths:
            files = listed_files(args.paths, repo_root)
            hits = path_hits([shown for shown, _ in files]) + scan_files(files, rules)
        else:
            hits = path_hits(staged_paths(repo_root)) + scan_diff(staged_diff(repo_root), rules)
    except CheckError as err:
        print("error: %s" % err, file=sys.stderr)
        return 2
    hits = sorted(set(hits))
    for path, line, rule in hits:
        print("%s:%d %s" % (path, line, rule), file=out)
    if hits:
        print("%d problem%s: fix them before committing" % (len(hits), "" if len(hits) == 1 else "s"), file=out)
        return 1
    print("no secrets found", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(run())
