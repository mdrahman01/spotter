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

## 2026-10-04 — Week 1: Cosmos Reason probe on a Nebius L40S endpoint

One attempt to serve `nvidia/Cosmos-Reason2-8B` with vLLM
(`vllm/vllm-openai:v0.18.0-cu130`) on a Serverless AI endpoint: `gpu-l40s-a`,
`1gpu-8vcpu-32gb`, on demand, no public IP, token auth.

**Startup stage times**

| Stage | Time |
| --- | --- |
| `nebius ai endpoint create --async` accepted | 6 s |
| `PROVISIONING` | 63 s |
| `STARTING` | 32 s |
| `IMAGE_PULLING` | 178 s |
| `RUNNING` with an https URL | reached 4 min 40 s after starting |
| Model download and load | did not happen (see errors) |
| `nebius ai endpoint delete` | 99 s |

States were sampled every 10 s. The endpoint existed for 6.7 minutes, about
$0.18 at $1.59 an hour.

**Errors hit**

- The model never loaded, for a reason that has nothing to do with Nebius or the
  GPU: vLLM exited at start-up with a Hugging Face `GatedRepoError` (HTTP 403),
  because the Hugging Face account behind our token had not been granted access
  to `nvidia/Cosmos-Reason2-8B`. The same token also gets 403 for
  `nvidia/Cosmos-Reason2-2B`, so we did not retry with the smaller model.
- The endpoint went from `RUNNING` to `ERROR` 23 s after it started. We found
  the cause in `nebius ai endpoint logs`.
- So whether the 8B model fits and runs on one L40S is still unanswered, and
  the three image tests did not run.

**Confusing in the Nebius CLI or docs**

- The installer puts the CLI in `~/.nebius/bin` and adds it to `PATH` in
  `.zshrc` only, so a script or any non-interactive shell gets
  `command not found: nebius`. Our scripts fall back to the full path.
- `nebius ai endpoint create --help` gives example platform and preset names
  but does not say how to list the valid ones. `--dry-run` was how we checked
  `gpu-l40s-a` and `1gpu-8vcpu-32gb` before paying for anything.
- The help for `get` does not describe the status fields or the possible
  states. We read them from the public API definition (`nebius/api` on GitHub):
  `PROVISIONING`, `STARTING`, `IMAGE_PULLING`, `RUNNING`, `STOPPING`, `STOPPED`,
  `DELETING`, `ERROR`.
- `nebius ai endpoint list --format json` prints `{}` when there are no
  endpoints, not `{"items": []}`, so a script has to allow for the missing key.
- The help marks `--parent-id` as required for `get-by-name`, while `create`
  and `list` take it from the CLI profile. We used `list` and filtered by name
  instead.
- `--token` and `--env KEY=VALUE` take secrets only as command-line arguments;
  the alternative is a MysteryBox secret. For a short probe, reading them from
  an environment variable or a file would be simpler and would keep them out of
  the process list.
- We did not open the docs site for this probe: the CLI help and the API
  definition were enough.

## 2026-10-04 — Cosmos Reason probe, second attempt

Same endpoint settings, after access to the gated model was granted on Hugging
Face. `scripts/cosmos_up.py` now checks that access first and creates nothing
without it.

**Startup stage times**

| Stage | Time |
| --- | --- |
| Access check and `nebius ai endpoint create --async` accepted | 10 s |
| `PROVISIONING` | 62 s |
| `STARTING` | 42 s |
| `IMAGE_PULLING` | 169 s |
| `STARTING` again | 10 s |
| `RUNNING` until `/v1/models` returned 200 (model download and load) | 337 s |
| Total, from create to model loaded | 630 s (10.5 min) |
| `nebius ai endpoint delete` | 101 s |

The endpoint existed for 12.4 minutes, about $0.33 at $1.59 an hour. With the
first attempt, the probe has cost about $0.51.

**What worked**

- `nvidia/Cosmos-Reason2-8B` loads and serves on one L40S (`gpu-l40s-a`,
  `1gpu-8vcpu-32gb`) with `--max-model-len 16384`. No retry with the 2B model
  was needed.
- Three requests with one image each (960x1280 or 1280x683 JPEG, about 900 to
  1,260 prompt tokens) took 5.2 s, 1.3 s and 1.9 s.

**Errors hit**

- All three tests were scored FAIL, because every response had empty
  `content`: the model's whole answer came back in the reasoning field. We
  served with `--reasoning-parser qwen3` and our prompts do not ask for
  `<think>` reasoning; the parser appears to treat output with no closing
  `</think>` tag as reasoning. Read from that field, the model was right about
  whether a vehicle was present in both images.
- One of the two JSON answers was not valid JSON (a missing comma), so a strict
  parser would have rejected it even in the right field.

**Confusing in the Nebius CLI or docs**

- `RUNNING` means the container has started, not that the model is serving:
  the managed URL answered 503 for another 5.6 minutes while vLLM downloaded
  and loaded the model. Polling `/v1/models` was the only readiness signal we
  found.
- The state went `STARTING`, `IMAGE_PULLING`, `STARTING`, `RUNNING`. The API
  definition describes `STARTING -> IMAGE_PULLING -> RUNNING`, so a script that
  reads a return to `STARTING` as a restart would be wrong.

## 2026-10-06 — Week 2: Nemotron as the attendant, calling tools

`nvidia/Nemotron-3_5-Lightning` on Nebius Token Factory decides what to do
about each ARRIVED or LEFT event by calling six tools: booking lookup, start
and end of a stay, bill, light, alert. Two replays of the 30 s test video, one
with a booking for the car and one without. Both scenarios passed on the first
run: 17 model calls, 14 tool calls, no tool errors.

**Latency**

- 0.30 to 0.71 s per model call, 0.40 s on average, with 1.2k to 1.8k prompt
  tokens each (system prompt, event, six tool schemas and the conversation so
  far).
- From an event to the light changing: 1.5 s (green on arrival) and 1.9 s (off
  on leaving) with a booking; 1.0 s and 0.6 s without. Each light change needs
  two or three model calls, because the model looks up the booking first.

**Reasoning tokens**

- Zero in all 17 calls. With tools in the request the model did not reason at
  all, where in Week 1 the same model spent 293 reasoning tokens on a one-line
  reply with no tools. Completion tokens were 11 to 74 per call.

**What it got wrong**

- Nothing against the rules: it looked up the booking first every time,
  started and ended the stay, billed, set the light, alerted the right person
  and finished with one sentence.
- It mostly made one tool call per round, so the LEFT event with a booking took
  all six rounds we allow (lookup, end, bill, alert, light off, final
  sentence); one more required step would have been cut off. For the unknown
  car it did put the red light and the owner alert in one round.
- The bill alert quoted the rate as "500 cents/hour", straight from the tool
  result, rather than as dollars.
