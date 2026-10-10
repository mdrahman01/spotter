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

## 2026-10-06 — Week 2: attendant fixes, alerts over Telegram

Same two scenarios as above, now with the owner and driver alerts delivered to
a Telegram chat as photos, a round cap of 10, a $1.00 minimum charge, dollar
amounts in tool results, and a tighter prompt for the unbooked car. Both
passed on the first run: 17 model calls, 15 tool calls, no tool errors, four
alerts delivered.

**Nemotron notes**

- Latency 0.30 to 0.64 s per call, 0.40 to 0.45 s on average, with 1.3k to
  1.9k prompt tokens; zero reasoning tokens again in every call. From an event
  to the light: 1.5 s and 2.9 s with a booking, 1.1 s and 1.6 s without (the
  Telegram upload now sits between some of those calls).
- The tightened rules were followed to the letter: the owner alert for the
  unbooked car says the car is in the spot and a photo is attached, with no
  advice to contact the driver; on leaving, the owner is told the car has gone
  and that it stayed 16 seconds, the figure the event carried.
- Giving the model "$5.00/hour" instead of a cents figure fixed the bill
  wording: the driver's bill alert now reads in dollars.
- It still made one tool call per round for the booked car (6 rounds for
  LEFT), but paired the alert and the light change in one round for the
  unbooked car, both on arrival and on leaving.
- Nothing it got wrong. It puts the plate text in alert messages, which is
  right for the owner and is masked in printed output by --expect.

## 2026-10-06 — Week 3: overstay warning, fee and owner alert

Two timer events now come from the watcher's clock: ENDING_SOON when the
booking ends within ending_soon_s, OVERSTAY once the car is still there
overstay_grace_s after the end, each once per stay. The code marks the stay
overstayed and adds a flat fee to the bill; the model only tells people. The
overstay scenario on the 30 s video (booking ending 12 s in, warning at 4 s,
grace 3 s) gave ARRIVED 6.6 s, ENDING_SOON 8.0 s, OVERSTAY 15.0 s, LEFT 26.2 s,
and a $6.00 bill ($1.00 parking at the minimum charge plus the $5.00 fee).
Overstay (Telegram alerts), booked and unknown (console alerts) all passed on
the first run.

**Nemotron notes**

- 34 model calls across the three scenarios, 0.29 to 0.65 s each, zero
  reasoning tokens throughout; prompt size now 1.4k to 2.1k tokens with eight
  rules and six tools. From OVERSTAY to the amber light: 1.0 s.
- On OVERSTAY it made three tool calls in one round (amber light, driver
  alert, owner alert), its most parallel round so far, and the bill alert
  after an overstay gave the breakdown exactly as asked.
- What it got wrong, all wording, no wrong tool call: the final sentence for a
  booked car's LEFT twice called it "the unbooked car" (the stayed_for field
  and the unbooked-car rule sit next to each other in its context); the bill
  alert for the plain booked stay gave a breakdown with a $0.00 fee although
  the rule asks for one only when a fee applied; and with 4 s left it said the
  booking "ends in 1 minute", which is the minutes_left=1 we handed it.
- One run of three ended with a native crash at interpreter exit (libc++
  "recursive_mutex lock failed", exit -6) after all output and the database
  writes were complete; it is from the ONNX/OpenCV teardown, not the model.

## 2026-10-07 — Week 3 fix-up: what caused Nemotron's wording slips

Fixes were made in what the model is given, not by adding rules. Over this
round it made 1135 calls in 99 saved runs, 0.27 to 0.64 s each, with
reasoning tokens in 0 of them.

- "The unbooked car has left", closing a booked car's LEFT. Two causes: every
  LEFT event carried stayed_for, a field only the unbooked-car rule mentions,
  so the model reached for that rule's sentence; and at LEFT the booking had
  expired, so lookup_booking answered outside_window, which the arrival rule
  equates with an unbooked car. Fixed by sending stayed_for only when there is
  no open stay, by reporting an open stay past its booking as "overstaying",
  and by taking the phrase out of the unbooked-car LEFT rule so there is no
  template to echo; the closing-sentence rule now asks for the actions taken.
  No alert or sentence said "unbooked" wrongly after that, in 10 runs.
- "$0.00" fee in a plain bill: compute_bill had shown it. The result now
  carries overstay_fee only when one applied.
- "Ends in 1 minute" with 4 s left: we had sent minutes_left=1. The event now
  carries time-left words made by code ("less than a minute", "about N
  minutes"), and the model used them as given in every run.
- Fee amount missing from the OVERSTAY driver alert: the amount was in the
  event but the rule did not point at it; it does now, and every alert since
  states "$5.00".
- Still open: in 2 of the final 3 overstay runs the LEFT of the overstaying
  car got an extra owner alert ("has left after staying ...") and no light-off;
  the model mixed the unbooked-car LEFT rule into the booked one once the
  booking had expired. The third run and the Telegram run were clean. Left as
  it is rather than adding a rule.
- Not the model: one run in 20 died at interpreter exit (libc++
  recursive_mutex) after all work was done; releasing the capture and the ONNX
  objects did not help (4 in 20), ending with os._exit after flushing did
  (0 in 20).

## 2026-10-08 — Week 3: finish check in code, with Nemotron Lightning's clean / reminder / code counts

A finish check now runs after every event: code works out from the database,
the light and this event's alerts what the rules require, tells the model once
in plain words what is still missing (up to 3 more rounds), and does whatever is
still missing itself. The same path covers an unreachable model and a model that
runs out of rounds. New rule: when an overstayed car leaves, the owner is told
it has gone and what was billed. compute_bill shows the model the total only
when no fee applied.

**Counts for nvidia/Nemotron-3_5-Lightning**, 12 runs, all PASS: 11 with console
alerts (overstay x5, booked x3, unknown x3) and one overstay run with Telegram
alerts.

- Events: 36 across the 12 runs: 30 clean, 6 after a reminder, 0 completed by
  code. Per run: 6 clean, 6 after a reminder, 0 completed by code (console runs
  alone: 5 clean, 6 after a reminder; the Telegram run was clean).
- By event: ARRIVED 12 clean; ENDING_SOON 6 clean; OVERSTAY 6 clean; LEFT 6
  clean, 6 after a reminder.
- Every reminder was about the same thing: on LEFT with an open stay the model
  stopped after the bill alert, one tool call short of set_light off. Told so,
  it set the light at once every time, so code never had to act.
- The new owner alert after an overstay was delivered with the bill total in
  all six overstay runs. The plain bill alert now reads with the total only.
- One wording slip seen: an owner alert after an overstay said the car had
  stayed "16 minutes" when it was 16 seconds; the model worked the duration out
  from the timestamps itself, and no check reads that figure.
- 167 model calls, 0.40 s on average, 0.71 s at most; reasoning tokens in 0 runs.

## 2026-10-09 — Week 4: the "16 minutes" slip and its cure; live source, bulb and calibration added

**The slip.** After an overstay, Nemotron Lightning told the owner the car had
stayed "16 minutes". It had stayed 16 seconds. The event gave the model raw
timestamps (arrived 04:30:00, left 04:30:16) and it worked the duration out
itself, getting the unit wrong. The same thing had already happened once with
the time left before a booking ends ("1 minute" for 4 seconds).

**The cure, same as for the time left.** The model is never given a raw
timestamp or a number of seconds again. Every clock time and duration it may
repeat is made by code, as words in the machine's local time zone: "6:02 pm",
"Sat 10 Oct, 11:56 am", "16 seconds", "1 minute 30 seconds", "1 hour 46
minutes". That covers the booking's start and end, the arrival, the last read,
the length of the stay, the time left and the time billed. The rules say to use
the words exactly as given and never to work out a time or duration. Code keeps
a list of the words it supplied for each event, and the scenario check fails
any alert that states a clock time or duration not on that list.

**Evidence this round.** In the two live runs against a local RTSP stream the
alerts said "16 seconds" and "34 seconds" for the stays and "1 minute" for the
time billed, all words code had supplied; the three replay scenarios passed
with 8 of 8 events clean, and the booked and overstay checks now include the
supplied-words rule. Latency and reasoning tokens unchanged: 0.3 to 0.7 s per
call, zero reasoning tokens.

**Also this round, no model change.** A live source (an RTSP camera, or the
sample video played in real time) with a newest-frame reader, watched time that
ignores gaps over 2 s, camera outage events with one owner alert each way from
code, a clean Ctrl-C; the Kasa bulb as the signal light with a console
stand-in; and python -m spotter.calibrate, which proposed a watch zone from the
sample video that gave the same ARRIVED and LEFT as the hand-set one.
