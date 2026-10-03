# Feedback

Dated notes on using NVIDIA Nemotron through Nebius Token Factory for Spotter.

## 2026-10-03 — Week 1: Nemotron hello and tool-calling test

**How long setup took**

- About 6 minutes from an empty repo to a working environment: a Python 3.13.2
  virtual env, `openai` 3.24.0 and `python-dotenv` 1.2.4 installed, and the key
  confirmed present in `.env`. The installs themselves were quick (1.2 s to
  create the env, 2.6 s for `pip install -r requirements.txt`); the rest was
  reading the repo and writing the files.
- The first run of `scripts/hello_nemotron.py` came about 13 minutes after
  starting, and passed.
- Not counted: creating the Nebius account and API key, which was done before
  this session.

**Errors hit**

- None. Both model IDs, `nvidia/Nemotron-3_5-Lightning` and
  `nvidia/Nemotron-3-Ultra-550b-a55b`, were accepted on the first call through
  the OpenAI SDK at `https://api.tokenfactory.nebius.com/v1/`. The script passed
  4 of 4 tests on its first run: for each model, an exact-reply test, and a
  `lookup_booking` tool call followed by a final answer built from the tool
  result.

**Confusing in the Nebius docs or console**

- Nothing to report on the docs or console yet: neither was opened in this
  session, because the base URL and model IDs we already had worked first time.
- One surprise from the API itself: both models reported reasoning tokens
  although the request did not ask for reasoning. The three-word reply
  "Spotter online." came back as 299 completion tokens on Lightning (293 of
  them reasoning) and 83 on Ultra (77 reasoning). We have not yet looked for
  how to turn reasoning off or cap it.
- The two models do not return quite the same response shape. Tool call IDs
  look like `call_...` on Lightning and `chatcmpl-tool-...` on Ultra, and
  `usage.prompt_tokens_details` is `null` on Lightning but an object on Ultra.
  Neither difference caused a problem with the OpenAI SDK.
