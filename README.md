# GoldAnalysisPipeline

Multi-agent pipeline for producing XAUUSD ("Nhận định Vàng") articles.

**This repository is at Round 7: the end-to-end orchestrator.** Round 1 turns a
raw human analysis plus OHLC data into an immutable **Run** with a
machine-readable `context.json`; Round 2 writes a Vietnamese XAUUSD commentary
from it; Round 3 audits that commentary and records a verdict; Round 4 applies
the audit; Round 5 decides deterministically whether the result may be
published; Round 6 sends it; Round 7 runs the whole sequence from one command.

Every stage is still a separate command and still works on its own. The
orchestrator coordinates those stages; **it does not bypass any gate.** Nothing
runs on a timer — schedulers are Round 9. See [Scope](#scope) for the full
boundary.

## What this is

The finished system will look like this:

```
Telegram Gold Bot / raw analysis  +  OHLC market data
        -> Normalize / Context Builder      <- Round 1
        -> Claude Writer                    <- Round 2
        -> ChatGPT Reviewer                 <- Round 3
        -> Claude Finalizer                 <- Round 4
        -> Deterministic Validator          <- Round 5
        -> Telegram Publisher               <- Round 6
        (all of the above, from one command) <- Round 7
```

What exists today:

```
raw analysis JSON  +  OHLC JSON
        -> validate -> normalize -> immutable Run -> context.json    (Round 1)
        -> versioned prompt -> Claude -> claude_draft.md
                                      -> claude_writer.json          (Round 2)
        -> verify hashes -> deterministic prechecks
        -> versioned prompt -> ChatGPT -> gpt_review.json            (Round 3)
        -> verdict routing -> Claude (only if revision needed)
                           -> claude_final.md
                           -> claude_finalizer.json                  (Round 4)
        -> verify chain -> deterministic checks
                        -> publish_decision.json  APPROVED|BLOCKED   (Round 5)
        -> re-verify approval -> publish_intent.json
                              -> Telegram sendMessage
                              -> publish_result.json                 (Round 6)
        -> one command drives all of the above, stopping wherever a
           stage declines                                            (Round 7)
```

`context.json` is the contract throughout. The writer receives it and nothing
else; the reviewer treats it as the highest authority and judges the article
against it; the finalizer edits the article to match it.

## How a Run works

Every execution creates a new Run directory. Runs are never reused and source
data is never rewritten.

1. **Generate a run id** — `20260828_022701_a83f2c`: UTC timestamp plus 24 bits
   of randomness. Sortable, readable, collision-resistant.
2. **Create the directory** — fails immediately if it already exists.
3. **Write `manifest.json`** with status `CREATED`.
4. **Capture the sources verbatim** — the exact payloads that were read, before
   any judgement is passed on them.
5. **Normalize and validate** — see [Validation policy](#validation-policy).
6. **Build and write `context.json`**.
7. **Flip the manifest to `NORMALIZED`**.

Step 4 happens *before* validation on purpose. When a Run fails, the input that
caused the failure is the most useful thing to keep. The safety property still
holds: **a failed Run ends at status `FAILED` and has no `context.json`**, so
nothing downstream can mistake it for usable input.

### Immutability

| File | Mutability |
| --- | --- |
| `telegram_input.json` | write-once |
| `ohlc.json` | write-once |
| `context.json` | write-once |
| `manifest.json` | rewritten per stage — it is the Run's ledger |

The storage layer refuses to overwrite a source or artifact that already exists.
Every file is recorded in the manifest with its SHA-256 and byte size, so
tampering is detectable after the fact. All writes are atomic (temp file →
`fsync` → `os.replace`), so a crash can never leave a half-written JSON document
that a later stage would happily parse.

## Quick start

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

Then create a Run from the shipped fixtures:

```bash
.venv/Scripts/python -m goldpipeline create-run --telegram fixtures/telegram_sample.json --ohlc fixtures/ohlc_sample.json --symbol XAUUSD --now 2026-08-28T02:20:12Z
```

```
Run created: 20260828_022701_a83f2c
Status: NORMALIZED
Context: runs\20260828_022701_a83f2c\context.json
Data quality: OK (20 bars)
```

Without installing, set `PYTHONPATH=src` instead of `pip install -e .`.

### Commands

| Command | Purpose |
| --- | --- |
| `create-run` | Normalize an analysis + OHLC pair into a new Run |
| `write-draft --run-id <id>` | Run the Claude Writer over a normalized Run |
| `review-draft --run-id <id>` | Run the ChatGPT Reviewer over a drafted Run |
| `finalize --run-id <id>` | Apply the review and store the final article |
| `gate-publish --run-id <id>` | Decide deterministically whether it may be published |
| `publish --run-id <id>` | Send an approved Run to Telegram, once |
| `pipeline-run` | Create a Run from JSON inputs and drive the whole pipeline |
| `pipeline-resume --run-id <id>` | Continue an existing Run from the stage it is due for |
| `show-run <run_id>` | Print a Run's status, files and digests |
| `list-runs` | List Run ids under the runs directory |

Useful `create-run` flags:

- `--symbol XAUUSD` — the instrument you expect. A mismatch fails the Run.
- `--dry-run` — validate the inputs and report, without creating a Run.
- `--now 2026-08-28T02:20:12Z` — pin the clock. The shipped fixtures have fixed
  timestamps, so without this the recency check will eventually report
  `STALE_DATA` (or `FUTURE_DATA`, depending on the date you run them). That is
  the check working, not a defect.
- `--json` — machine-readable output.

Exit codes: `0` success, `1` configuration problem, `2` unusable data.

## The writer stage

```bash
python -m goldpipeline write-draft --run-id 20260828_022701_a83f2c --fake-writer
```

```
Run: 20260828_022701_a83f2c
Writer: fake (fake-writer-v1)
Status: COMPLETED
Draft: runs/20260828_022701_a83f2c/claude_draft.md
Metadata: runs/20260828_022701_a83f2c/claude_writer.json
Article: 489 chars, 2 claims
```

Drop `--fake-writer` to call Claude for real. That needs `ANTHROPIC_API_KEY` in
the environment; `ANTHROPIC_MODEL` (or `--model`) selects the model, defaulting
to `claude-opus-5`. See `.env.example` for the full list.

Exit codes: `0` success, `1` configuration problem (missing key, unknown model),
`2` the data or the provider answer was unusable.

### What the writer may and may not do

It may rewrite, reorganise and shorten the analyst's note, describe price
behaviour from the candles, and carry the analyst's view through as a view.

It may not invent a price, a high, a low, a timestamp, an indicator value, or a
news event; change the symbol or timeframe; or produce a BUY/SELL call the
source did not make. **If the data does not support a claim, the claim is
omitted** - the prompt states this as a rule, and the checks below back it up.

### Artifacts

| File | Contents |
| --- | --- |
| `claude_draft.md` | The article, and nothing else - no reasoning, no prompt, no metadata |
| `claude_writer.json` | Run id, status, model, provider, prompt version, claims, warnings, usage |

The article is stored once. `claude_writer.json` carries `article_sha256` and
`article_chars` instead of a copy, which binds the two files together: if either
changes, they stop agreeing.

`source_claims` is what makes a draft auditable - each entry records a number or
attributed view the article used, and where in the context it came from:

```json
{
  "type": "PRICE",
  "value": "3314.20",
  "source": "context.price.latest_close"
}
```

### Prompt

The prompt is versioned in `src/goldpipeline/prompts/` (`gold_writer_v1.md`) and
the version used is recorded on every artifact. It has four sections:
`SYSTEM RULES` and `OUTPUT CONTRACT` in the system turn, `MARKET FACTS` and
`UNTRUSTED SOURCE DATA` in the user turn. A structural test fails if a section
disappears.

### Safety rules

**The analyst's note can never reach the model as an instruction.** Three
mechanisms enforce it:

1. *Channel separation* - the system turn is the template on disk, byte for
   byte. No source data is interpolated into it, not even the symbol.
2. *Nonce delimiters* - the note is fenced between markers containing a random
   per-request token. Text trying to close the block early would have to guess
   it; a fixed delimiter could simply be typed by the source.
3. *A restated contract* - the user turn names the fence and says plainly that
   everything inside it is data.

`fixtures/telegram_injection.json` is an adversarial message that tries to change
the symbol, print the API key, and assert a false price. It is covered by tests
end to end.

**Prices the market data denies are caught before drafting.** Prices mentioned in
the note are extracted in Python and compared against the candle range; a number
outside it becomes a `SOURCE_PRICE_OUT_OF_RANGE` warning *and* an explicit
caution in the prompt. The band is 0.5% of price (or 25% of the observed range,
whichever is wider), so an analyst naming a resistance just above the session
high is not flagged, while "gia hien tai 3400" against a 3305-3322 window is.
The note itself is never edited.

**Nothing is written until everything succeeds.** The provider is called, the
answer is validated, both artifacts are serialized in memory, and only then are
they committed as one all-or-nothing unit. A failure at any earlier point leaves
the Run byte-for-byte as Round 1 produced it, plus one failure event on the
manifest ledger - so the stage can simply be run again.

**A drafted Run is never overwritten.** Re-running `write-draft` on a Run that
already has artifacts fails with `WRITER_ARTIFACT_EXISTS`, before the provider is
called. Retrying after a *failure* is allowed, because no artifacts exist.

**Credentials never leave the environment.** The key is read in `config.py`, held
behind a `__repr__` that redacts it, and passed only to the SDK client. Provider
error bodies are never echoed into messages that reach the manifest.

### Response validation

A provider answer is rejected - and no draft written - when it is invalid against
the schema, has no structured output, echoes the wrong `run_id`, carries an empty
or stub article, was truncated at the token limit, or was a refusal. There is no
regex rescue of a malformed answer: a failed draft is better than a
plausible-looking one assembled from fragments.

## The reviewer stage

```bash
python -m goldpipeline review-draft --run-id 20260828_022701_a83f2c --fake-reviewer
```

```
Run: 20260828_022701_a83f2c
Reviewer: fake (fake-reviewer-v1)
Verdict: PASS
Score: 95
Issues: 0
Review: runs/20260828_022701_a83f2c/gpt_review.json
```

Drop `--fake-reviewer` to call OpenAI for real. That needs `OPENAI_API_KEY` in
the environment; `OPENAI_REVIEW_MODEL` (or `--model`) selects the model,
defaulting to `gpt-5.1`.

Exit codes match the writer: `0` success, `1` configuration problem, `2` the
data or the provider answer was unusable.

### What the reviewer does and does not do

It checks data accuracy against the context, hunts unsupported claims, watches
for the analyst's view being inflated into certainty, looks for internal
contradictions and absolute risk language, and flags style only when it really
hurts the piece.

**It never rewrites the article.** There is no schema field for a corrected
version, `extra="forbid"` rejects an invented one, and instructions are length-
capped so a draft cannot be smuggled through as a "suggestion". It returns
issues and revision instructions; correcting the text is Round 4's job.

### Verdicts

| Verdict | Meaning |
| --- | --- |
| `PASS` | Nothing worth fixing. Requires no HIGH/CRITICAL issue and score ≥ 90. |
| `NEEDS_REVISION` | Usable once specific things are corrected. The normal outcome for a wrong number. |
| `REJECT` | A critical factual error, the wrong instrument, or a piece small edits cannot save. |

A provider or network failure is **not** a verdict — it says nothing about the
article, so the stage fails rather than recording a `REJECT`.

`RunStatus.REVIEWED` means a review happened, not that it passed. A Run may
legitimately be `REVIEWED` with a verdict of `REJECT`; the verdict lives in the
artifact.

### Deterministic prechecks

Before any API call, Python checks what it can prove:

| Check | Finding | Severity |
| --- | --- | --- |
| Artifact digests vs the manifest, and the writer's own cross-references | stage fails, **no API call** | — |
| A `source_claims` path that does not resolve | `CLAIM_SOURCE_NOT_FOUND` | HIGH |
| A claim whose value disagrees with the context | `CLAIM_VALUE_MISMATCH` | HIGH |
| A price-like number no context value, claim, or in-range analyst level explains | `UNKNOWN_PRICE_LIKE_NUMBER` | MEDIUM |
| …and far outside the candle range | `NUMBER_OUTSIDE_MARKET_RANGE` | HIGH |
| A foreign instrument in an XAUUSD article | `FOREIGN_SYMBOL_MENTIONED` | CRITICAL |
| RSI, MACD and friends — the context carries none | `UNSUPPORTED_INDICATOR_MENTIONED` | HIGH |
| "chắc chắn", "không thể", "đảm bảo" | `ABSOLUTE_RISK_LANGUAGE` | HIGH |

Claim paths are resolved by a whitelist walker — declared Pydantic fields and
integer list indices only, no `eval`, no dunders, no methods, capped depth. A
path a model invented cannot become code.

The numeric scanner is deliberately conservative: `M15`, `H1`, `24 giờ`,
`2 kịch bản`, `0.5 lot`, `2%` and four-digit years are not prices, and a number
the analyst named near the market is allowed through. Out-of-range numbers in
the note are *not* laundered — an injected 9999 stays flagged wherever it came
from.

### Two kinds of disagreement

The pipeline does not simply trust the reviewer, and it distinguishes two cases:

**The response contradicts itself** — a `PASS` listing a HIGH issue, a
`NEEDS_REVISION` with nothing to revise, a factual issue with no evidence, an
answer about a different Run. These are broken answers, not lenient ones. The
response is rejected and no artifact is written.

**The response is more generous than the evidence allows** — the prechecks found
a wrong claim or a foreign instrument and the reviewer still passed it. Here the
verdict is escalated (HIGH → `NEEDS_REVISION`, CRITICAL → `REJECT`), the missed
findings become issues, and the artifact records `verdict_source:
POLICY_ESCALATED` with a note saying why. Escalating rather than rejecting keeps
the evidence Round 4 needs; nothing is silent.

Policy only ever escalates — it never talks a `REJECT` down.

### Safety

The article is untrusted content too. Both it and the analyst's note are fenced
in the user turn with separate nonce-bearing labels, and the system turn is the
on-disk template byte for byte — no data is interpolated into it, ever. An
article that asks to be passed is itself a `PROMPT_INJECTION` issue.
`fixtures/article_injection.md` exercises this end to end.

### `gpt_review.json`

Verdict, score, summary, issues (with evidence), revision instructions, the
model's own verdict before policy, the deterministic findings, prompt version,
usage, and the SHA-256 of all three inputs — so a later stage can prove the
review describes the artifacts it is looking at. No chain of thought, no system
prompt.

## The finalizer stage

```bash
python -m goldpipeline finalize --run-id 20260828_022701_a83f2c --fake-finalizer
```

The verdict decides the path, and two of the three never reach a provider:

| Verdict | What happens | Provider called | API key needed |
| --- | --- | --- | --- |
| `PASS` | The draft becomes the final article, byte for byte | No | No |
| `NEEDS_REVISION` | Claude applies the review's corrections | Yes | Yes |
| `REJECT` | Blocked. The Run waits for a human | No | No |

```
Run: 20260828_022701_a83f2c
Review: PASS
Finalization: PASSTHROUGH
Provider called: No
Final: runs/20260828_022701_a83f2c/claude_final.md
```

Exit codes: `0` success, `1` configuration problem, `2` unusable data or a
rejected revision, `3` **blocked** — a distinct code because retrying will not
help; a human has to look at it.

Drop `--fake-finalizer` to call Claude for real. `ANTHROPIC_FINALIZER_MODEL`
selects the model, falling back to `ANTHROPIC_MODEL` and then to
`claude-opus-5`.

### Why PASS never calls a model

A passing article is already correct. Asking a model to "improve" it spends
money to introduce drift into something that survived review — the opposite of
the job. So the draft's exact bytes become the final article: no whitespace
tidying, no trailing-newline normalization, no re-titling. `finalization_mode`
records `PASSTHROUGH` and `provider_called` records `false`.

### What the finalizer may and may not do

It is an **editor**, not a second analyst. It applies the review's corrections
and changes as little else as possible. If one price is wrong, it fixes that
price — it does not also reorder sections, swap icons or "improve the flow".

It may not introduce a price, an indicator value, a news event, a different
instrument, or a signal the original article did not make. If removing an
unsupported claim leaves a section thin, the section stays thin or goes.

### Issue resolutions

Every issue in `gpt_review.json` must be answered by `issue_id`:

| Resolution | When |
| --- | --- |
| `APPLIED` | The article was changed to address it |
| `NOT_APPLICABLE` | It does not hold. **LOW/MEDIUM only**, with a reason |
| `BLOCKED` | Real but not fixable by editing. **LOW/MEDIUM only** |

**A HIGH or CRITICAL issue must be `APPLIED`.** Letting a model mark a wrong
price "not applicable" would be the one escape hatch that makes the whole review
chain decorative, so a response that does is rejected outright.

### Deterministic postchecks

After the model answers and before anything is written:

1. **Every issue is accounted for** — no missing resolution, no duplicate, no
   `issue_id` the review never raised.
2. **Severe issues were actually applied** — see above.
3. **The article got better, not different.** The deterministic checks are
   re-run on the revision and compared against the draft. A HIGH/CRITICAL
   finding the original did not have is a regression; one that survived is a fix
   that did not happen. Either fails the revision.
4. **Claimed corrections are visible.** If an issue cites `evidence.actual` —
   the wrong value as it appeared — and the finalizer says it applied the fix,
   that value must be gone from the text.

Gate 3 is the one that earns its keep. A model told to remove an invented RSI
reading will comply and add an EMA200 in the same breath, fluently, and
invisibly to anything that only reads the resolutions.

`source_claims` are deliberately **not** re-checked here: they record what the
*writer* used and live in an artifact the finalizer may not rewrite, so a stale
claim would survive a correct revision and no edit could ever clear it.

### Artifacts

| File | Contents |
| --- | --- |
| `claude_final.md` | The final article, and nothing else |
| `claude_finalizer.json` | Mode, verdict, resolutions, postcheck findings, usage, and the digests of all four inputs |

### Safety

Three untrusted blocks reach this stage, not two: the analyst's note, the
article, and **the review** — which is another model's output and whose evidence
fields quote the article back verbatim. All three are fenced with a per-request
nonce under distinct labels, and the system turn is the on-disk template byte for
byte.

## The publish gate

```bash
python -m goldpipeline gate-publish --run-id 20260828_022701_a83f2c
```

```
Run: 20260828_022701_a83f2c
Gate: gold_publish_gate_v1
Decision: APPROVED
Checks: 16 passed, 0 warnings, 0 failed
Artifact: runs/20260828_022701_a83f2c/publish_decision.json
```

**No AI, no network, no credentials.** Every decision here is made by code. The
command runs with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` unset, and a test
asserts it opens no socket.

Exit codes: `0` approved, `3` blocked, `1` configuration problem, `2` the Run is
not gateable or its decision already exists. A block is **not** a crash — it is
the gate working.

### The contract for Round 6

> **The publisher MUST NOT publish unless `publish_decision.decision == APPROVED`
> and the digests recorded in that decision still match the artifacts on disk.**

The decision names the exact bytes it approved. If `claude_final.md` changed
after the gate ran, the approval no longer describes it.

### What it decides

One question — *is it safe to publish this automatically?* — and two answers,
`APPROVED` or `BLOCKED`. The gate never edits, sanitises or retries. An article
it will not approve needs a human or a new Run; quietly cleaning one up would
make the boundary decorative.

**Fail closed.** Anything uncertain about credentials, model-control prose, a
foreign instrument, an unsupported indicator or a suspicious price blocks. A
false block costs someone a look; a false approval publishes it.

### The checks

| Check | Blocks on |
| --- | --- |
| `ARTIFACT_CHAIN_INTEGRITY` | Any of the 8 artifacts failing its manifest digest, or a cross-reference disagreeing |
| `RUN_STATE` | The Run is not `FINALIZED` |
| `REVIEW_VERDICT_STATE` | A `REJECT` review that somehow finalized, or a finalization that disagrees with the review |
| `REVIEW_ISSUE_CLOSURE` | An unanswered issue, or a HIGH/CRITICAL one not `APPLIED` |
| `CORRECTION_CLOSURE` | A value the review called wrong still in the text |
| `ARTICLE_STRUCTURE` | Empty, too short, too long, a JSON dump, a traceback, a code fence |
| `TELEGRAM_COMPATIBILITY` | Bad UTF-8 or control characters (over-length is a warning) |
| `INSTRUCTION_SHAPED_TEXT` | Model-control prose, English or Vietnamese |
| `CREDENTIAL_EXPOSURE` | A credential-shaped value |
| `FOREIGN_SYMBOL` | An instrument other than the Run's |
| `UNSUPPORTED_INDICATOR` | RSI, EMA200, MACD… — the context carries none |
| `SUSPICIOUS_PRICE` | A price-like number the data does not account for |
| `RISK_LANGUAGE` | "chắc chắn", "không thể", guarantees |
| `EXTERNAL_FACT_WITHOUT_SOURCE` | A claim that a named economic event occurred |
| `CONTEXT_CONSISTENCY` | A timeframe other than the Run's |
| `NO_NEW_REGRESSION` | A severe problem the draft did not have |

The symbol, indicator, price and risk-language scanners are Round 3's, reused
rather than restated — one definition of "foreign symbol" for the whole
pipeline. **The gate rates one of them more severely than the review does:** an
unexplained price-like number is MEDIUM mid-pipeline, where a later stage may
still fix it, and HIGH here, where there is no later stage.

### Instruction-shaped text

Round 4 leaves a gap by design: a finalizer making the *minimum necessary* edit
leaves "Ignore all previous instructions" in the prose, because no review issue
named it. This gate is where that is caught — in both languages, and matched on
diacritic-folded text so `bỏ qua`, `bo qua` and `BỎ QUA` all hit.

### Credential redaction

A detected token is **never** copied into `publish_decision.json`. The finding
records its shape and a redaction that keeps the vendor prefix and the last four
characters — enough to know which key to rotate, not enough to use it:

```
sk-pro…wxyz (<redacted:38 chars>)
```

### Integrity failures

A tampered-but-readable Run gets a `BLOCKED` **decision** with a generic
`ARTIFACT_INTEGRITY_FAILURE` blocker; the tampered content is never quoted back
as evidence, and no later check runs on it. A Run whose *manifest* cannot be
parsed raises instead — there is no trustworthy identity to attach a decision to,
so writing one would be inventing provenance.

### Decisions are immutable

Re-running the gate on a Run that already has a decision is refused. There is no
`--force`: re-evaluating under a newer gate would let an approval appear where a
block used to be, with nothing recording that it changed.

## The publisher

```bash
python -m goldpipeline publish --run-id 20260828_022701_a83f2c --fake-publisher
```

```
Run: 20260828_022701_a83f2c
Publisher: fake -> @fake_offline_channel
Status: PUBLISHED
Delivered: 1/1 message(s)
  chunk 0: message_id=1000
Artifact: runs/20260828_022701_a83f2c/publish_result.json
```

Drop `--fake-publisher` to post for real. That needs `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_TARGET_CHAT_ID` in the environment.

> **The implementation and its tests never publish for real.** The whole suite
> runs against the offline client, and a socket guard fails any test that tries
> to connect. Nothing in this repository has posted to a Telegram channel.

Exit codes: `0` published, `3` any other delivery outcome, `1` configuration
problem, `2` the Run may not be published or was already attempted.

### Preconditions

The publisher sends only when **all** of these hold, and checks every one before
it writes anything:

- the Run is `READY_TO_PUBLISH`;
- `publish_decision.decision` is `APPROVED`;
- the decision came from a gate version this publisher supports;
- `claude_final.md`, `claude_finalizer.json`, `context.json` and
  `publish_decision.json` still match the digests recorded for them;
- no `publish_result.json` exists — one attempt per Run;
- no orphaned `publish_intent.json` exists.

The article is then read **once** and held in memory. Chunking, hashing and
sending all work from that snapshot, so a file edited mid-publish cannot change
what goes out.

### Delivery outcomes

| Status | Meaning |
| --- | --- |
| `PUBLISHED` | Every chunk confirmed by Telegram with a message id |
| `FAILED` | Nothing delivered, and Telegram said so explicitly |
| `PARTIAL` | Some chunks confirmed, then an explicit refusal |
| `UNCERTAIN` | A timeout, reset, 5xx or unparseable reply — delivery unknown |

The distinction that matters is between *knowing* and *not knowing*. An explicit
refusal is good news: nothing was delivered, and that is certain. `UNCERTAIN`
means Telegram may or may not hold the message.

> **`UNCERTAIN` MUST NEVER BE AUTOMATICALLY RETRIED.** A duplicate post cannot be
> taken back — readers have already seen it. Reconcile by looking at the channel.

`UNCERTAIN` outranks `PARTIAL`: if the last chunk's fate is unknown, so is the
attempt's. Once any chunk is refused or unknown, the remaining chunks are not
sent.

### Retry policy

Exactly one condition is retried: an explicit **429** with a `retry_after`. That
is Telegram stating it did *not* accept the request, so a retry cannot duplicate.
The delay is honoured, capped at 300 seconds, and bounded to 2 retries per chunk.

Nothing else is retried — not timeouts, not connection resets, not 5xx, not a
malformed reply. Each of those may mean the message was delivered and the
acknowledgement lost.

### The durable intent

`publish_intent.json` is committed **before the first request** and records the
destination, the article digest and the chunk plan. It exists for one scenario:

```
POST sendMessage  ->  Telegram accepts and posts  ->  connection drops
                  ->  process sees a failure      ->  process dies
```

Without the intent, the next run cannot tell "never sent" from "sent,
acknowledgement lost", and a retry would post the article twice. With it, the
next run sees an intent and no result, records `UNCERTAIN`, and **sends nothing**.

That includes the unavoidable window where Telegram confirms and the process dies
before `publish_result.json` is written. The Run is then `PUBLISH_UNCERTAIN` and
needs a human. This is deliberate: the pipeline does not fake exactly-once
delivery by retrying blindly.

### Chunking

Telegram caps a message at 4096 UTF-16 code units; the publisher targets 3900.
Length is counted the way Telegram counts it, so an emoji above the BMP costs
two — a naive character count would let an emoji-heavy article through and have
it refused.

Cuts prefer a paragraph break, then a line break, then a sentence end, then any
whitespace, then a hard boundary. The invariant, asserted before sending:

```python
"".join(chunks) == article  # exactly
```

Nothing is stripped, normalised, or annotated. There is no `(1/2)` marker — that
would be content the gate never approved. Almost every article fits in one
message; multi-part sends are paced at 1.1 s between chunks.

### Security

The bot token appears in the Bot API **URL**, which makes every HTTP exception a
potential credential leak. No exception from the HTTP layer is allowed to
propagate: each is caught and replaced with a message written in this codebase,
and even the exception's `__cause__` is scrubbed so a printed traceback cannot
expose it. Tests assert a token sentinel appears in no artifact, log, exception
or CLI output.

The destination comes only from `TELEGRAM_TARGET_CHAT_ID`. There is deliberately
no `--chat-id` flag, and article content is never read for configuration — a
pipeline whose destination its own content could steer would be one prompt away
from posting to a stranger's channel.

Messages are sent as plain text with no `parse_mode`, so no markup parser can
restyle or swallow text nobody reviewed in that form.

## The orchestrator

```bash
python -m goldpipeline pipeline-run --telegram fixtures/telegram_sample.json --ohlc fixtures/ohlc_sample.json --symbol XAUUSD --fake-ai --now 2026-08-28T03:00:00Z
```

```
Run: 20260828_030114_9c1e2a
Mode: READY_FOR_PUBLISH

NORMALIZE  COMPLETED
WRITER     COMPLETED
REVIEW     PASS
FINALIZE   PASSTHROUGH
GATE       APPROVED

Final status: READY_TO_PUBLISH
Execution: COMPLETED
```

**The orchestrator coordinates the existing stages. It does not bypass any
gate.** It holds no business logic of its own: every verdict it reports was
reached by the stage that already owned that question. What it adds is
sequencing, a stop policy, and a concurrency guard.

### Modes

The mode is a *ceiling*, not a target — a Run that stops early because a gate
declined has still honoured its mode.

| Mode | Runs | Stops at |
| --- | --- | --- |
| `generate-only` | create → write → review → finalize | `FINALIZED` |
| `ready-for-publish` *(default)* | …plus the publish gate | `READY_TO_PUBLISH` or `PUBLISH_BLOCKED` |
| `publish` | …plus the publisher | `PUBLISHED` / `PUBLISH_FAILED` / `PARTIALLY_PUBLISHED` / `PUBLISH_UNCERTAIN` |

The default runs every check and sends nothing. Publishing is always a separate,
explicit decision:

```bash
python -m goldpipeline pipeline-run --telegram ... --ohlc ... --fake-ai --publish --fake-publisher
```

For the real transport, `--publish` alone is not enough — `--confirm-real-publish`
must be there too. There is deliberately no `--all` and no single flag that both
builds an article and posts it. Both flags name what they do, and neither can be
reached by accident. (`--fake-publisher` cannot reach anyone, so it needs no
confirmation; if both are given, the offline transport wins.)

### Resume

`pipeline-resume` continues a Run from whichever stage its status says is due.
Completed stages are not repeated, and their artifacts are not rewritten.

| Run status | Next stage |
| --- | --- |
| `NORMALIZED` | writer |
| `DRAFTED` | reviewer |
| `REVIEWED` | finalizer — unless the verdict was `REJECT` |
| `FINALIZED` | publish gate |
| `READY_TO_PUBLISH` | publisher, and only in `publish` mode |

Everything else is terminal:

| Run status | Result | Why |
| --- | --- | --- |
| `PUBLISHED` | `ALREADY_COMPLETED` | Done. Zero provider calls, and not an error. |
| `PUBLISH_BLOCKED` | `BLOCKED` | The decision is immutable. Re-judging needs a new Run. |
| `PUBLISHING` | `NOT_RESUMABLE` | A send may be in flight, or was interrupted. |
| `PUBLISH_UNCERTAIN` | `NOT_RESUMABLE` | **Never retried.** Telegram may hold the article. |
| `PARTIALLY_PUBLISHED` | `NOT_RESUMABLE` | Readers already saw part of it. |
| `PUBLISH_FAILED` | `NOT_RESUMABLE` | One attempt per Run. Fix the cause, create a new Run. |
| `FAILED` | `NOT_RESUMABLE` | The Run's own inputs were rejected. |

### Stop conditions

A **business stop** is a stage declining. Nothing broke, retrying will not help,
and the pipeline returns a result rather than raising:

- **review `REJECT`** — reported as stopping at `REVIEW`, not `FINALIZE`. Round 4
  would refuse the article anyway, but calling it in order to be refused would
  name the wrong stage as the one that ended the pipeline. The finalizer, gate
  and publisher are not called at all.
- **gate `BLOCKED`** — the publisher is not called.
- **publish `FAILED` / `PARTIAL` / `UNCERTAIN`** — the outcome is propagated with
  its own meaning intact. The orchestrator adds no retry of its own.

An **execution failure** — a provider timeout, a missing key, a tampered
artifact — ends the invocation with status `FAILED` and the stage's own error
type preserved, so a caller can still tell a provider outage from bad data.

### Lazy provider configuration

Clients are built from factories, called at most once and only when their stage
is about to run. A Run resumed at the gate needs no API keys at all; a Run
resumed at the publisher needs Telegram configuration and nothing else; and a
review that passes never builds a finalizer client, preserving Round 4's
zero-call passthrough.

### One lock per Run

An orchestrated Run holds `.pipeline.lock` inside its own directory for the
duration of the invocation, acquired with `O_CREAT | O_EXCL`. Two invocations on
*different* Runs never contend — there is no global lock.

**A stale lock is never removed automatically.** A lock left by a killed process
is indistinguishable from one held by a process that is mid-`sendMessage`, and
this pipeline's most expensive mistake is posting the same article twice. The
lock records its holder's pid, hostname and start time; clearing it is a
deliberate human act.

### No execution artifact

There is no `pipeline_execution.json`. A Run is legitimately orchestrated more
than once over its life — created today, resumed and published tomorrow — and
artifacts here are write-once, so such a file would have to either fail on the
second invocation or be overwritten. Both break the model the rest of the rounds
depend on. The audit trail lives in the manifest's event log (`RUN_CREATED`,
`WRITER_COMPLETED`, `REVIEW_COMPLETED`, `FINALIZER_COMPLETED`, `GATE_APPROVED` /
`GATE_BLOCKED`, `PUBLISH_COMPLETED`, `PIPELINE_STOPPED`, `PIPELINE_FAILED`),
beside the events the stages write for themselves; the machine-readable view is
`--json`.

### Exit codes

`0` completed or already complete · `1` configuration problem, a locked Run, or a
refused flag combination · `2` unusable data or a stage failure · `3` a gate
declined, or a delivery was not fully confirmed.

## Run directory structure

```
runs/
└── 20260828_022701_a83f2c/
    ├── manifest.json         # status, events, file digests
    ├── telegram_input.json   # source, verbatim          (Round 1)
    ├── ohlc.json             # source, verbatim          (Round 1)
    ├── context.json          # what the writer reads     (Round 1)
    ├── claude_draft.md       # the article               (Round 2)
    └── claude_writer.json    # draft metadata            (Round 2)
```

Later stages add `gpt_review.json` (Round 3), `claude_final.md` and
`claude_finalizer.json` (Round 4), `publish_decision.json` (Round 5), and
`publish_intent.json` and `publish_result.json` (Round 6) to the same directory.
No placeholders are created for them. An orchestrated Run also holds a transient
`.pipeline.lock` while it is being driven; it is not an artifact and is not
recorded in the manifest.

A Run's status moves `CREATED -> NORMALIZED -> DRAFTED -> REVIEWED -> FINALIZED
-> READY_TO_PUBLISH | PUBLISH_BLOCKED`, and from `READY_TO_PUBLISH` into the
publish-side outcomes. A failed *writer* stage leaves the Run at `NORMALIZED`
rather than `FAILED`: the inputs are still valid and the stage can be retried.
`FAILED` is reserved for a Run whose own inputs are unusable.

## Schemas

All schemas are Pydantic models under `src/goldpipeline/schemas/`.

| Schema | Role |
| --- | --- |
| `TelegramAnalysisInput` | Raw analysis message. Only `raw_text` is required; missing metadata is `null`, never invented. |
| `OHLCBar` | One candle. Enforces `low <= open,close <= high` and `high >= low` at construction. |
| `MarketDataInput` | Loose provider payload: derived fields optional, naive timestamps tolerated. |
| `MarketDataSnapshot` | Normalized market data. Every invariant is guaranteed — `latest_bar` **is** `bars[-1]`, bars are ascending, unique and UTC. |
| `AnalysisContext` | The document downstream agents read. Facts only. |
| `DataQuality` | Quality report embedded in every context. |
| `RunManifest` | Run ledger: status, events, source and artifact digests. |
| `ReviewModelOutput` | What the reviewer may author: verdict, score, issues, instructions. No field for a rewritten article. |
| `ReviewResult` | The `gpt_review.json` artifact, with the digests of all three inputs. |
| `PrecheckFinding` | One deterministic observation, made before any model was consulted. |
| `FinalizerModelOutput` | What the finalizer may author: a revised article and one resolution per issue. No verdict, no score. |
| `FinalizerResult` | The `claude_finalizer.json` artifact, with the digests of all four inputs. |
| `PublishDecision` | The `publish_decision.json` artifact: verdict, checks, blockers, and the digests of all six inputs. |
| `PublishIntent` | The `publish_intent.json` artifact, written before the first request. |
| `PublishResult` | The `publish_result.json` artifact: outcome, confirmed message ids, and why an attempt stopped. |
| `PipelineExecutionResult` | One orchestrator invocation: mode, per-stage outcomes, where it stopped. Returned, never stored. |

### Two things worth knowing about the data types

**Prices are `Decimal`, serialized as JSON strings** (`"3312.45"`). A JSON float
would introduce binary representation drift into numbers an agent quotes
verbatim into a published article. Trailing zeros are not padded — `3315.10`
becomes `"3315.1"` — because the pipeline does not invent precision. The value is
exact either way.

**Timestamps are timezone-aware UTC**, serialized as `2026-08-28T02:15:00Z`. A
naive timestamp is converted using the payload's declared `timezone`; if the
payload declares none, the Run **fails** rather than assuming UTC. Guessing here
would silently shift every candle by the provider's offset. The original
declaration is preserved as `source_timezone` for audit.

### `context.json` shape

```json
{
  "schema_version": "1.0.0",
  "run_id": "20260828_022701_a83f2c",
  "market":  { "symbol": "XAUUSD", "timeframe": "M15", "provider": "mt5-demo",
               "timezone": "UTC", "source_timezone": "UTC" },
  "timing":  { "generated_at": "...", "requested_at": "...", "data_from": "...",
               "data_to": "...", "latest_candle_at": "..." },
  "price":   { "latest_open": "3312.45", "latest_high": "3315.1",
               "latest_low": "3311.9",  "latest_close": "3314.2" },
  "raw_analysis": { "text": "...", "source": "telegram", "message_id": 48217,
                    "trust_level": "UNTRUSTED", "handling": "..." },
  "ohlc":    { "bar_count": 20, "bars": [ ... ] },
  "data_quality": { "bar_count": 20, "missing_fields": [], "warnings": [],
                    "status": "OK" }
}
```

`price` is derived from `ohlc.bars[-1]`, never copied from a provider field.

## Validation policy

The distinction that matters: a **validation error** means the data contradicts
itself and the Run fails. A **quality warning** means the data is usable but
degraded, and is recorded in `context.data_quality`.

| Situation | Policy |
| --- | --- |
| Broken OHLC ordering | **FAIL** |
| Duplicate bar timestamp | **FAIL** — never silently de-duplicated; the pipeline cannot know which bar is authoritative |
| Naive timestamp with no declared timezone | **FAIL** |
| Symbol disagreement (payload, per-bar, or `--symbol`) | **FAIL** |
| Provider `latest_bar` disagrees with `bars[-1]` | **FAIL** |
| Empty analysis text | **FAIL** |
| Bars out of order | sorted ascending + `BARS_REORDERED` |
| Missing volume | `MISSING_VOLUME` + `missing_fields` entry |
| Gaps in the series | `BAR_GAPS` (weekends and session breaks are normal) |
| Latest candle in the future | `FUTURE_DATA` — a clock or timezone bug; prices must not be quoted from a candle that has not closed |
| Latest candle very old | `STALE_DATA` |
| Missing optional message metadata | `MISSING_TELEGRAM_METADATA` |
| Invisible characters in the analysis text | stripped + `RAW_TEXT_SANITIZED` |

`data_quality.status` is `OK` when there are no warnings and `WARNING`
otherwise. A `FAIL` never appears in a stored context — a fatal problem stops
the Run before one is written.

## Untrusted input

`raw_analysis.text` is third-party content and is treated as **data, never as
instructions**. The context carries `trust_level: "UNTRUSTED"` and an explicit
handling note so a Round 2 prompt builder cannot forget it.

Round 1 enforces the mechanical side of this:

- no `eval`, no dynamic imports, no shell execution anywhere in the codebase;
- source payloads cannot alter configuration — every setting is a CLI argument
  or an environment variable;
- unknown JSON keys are rejected rather than absorbed;
- zero-width and bidirectional-override characters are stripped from the
  analysis text. They are invisible to a human reviewer but not to a model,
  which makes them a natural place to hide text. Wording, punctuation and line
  structure are otherwise preserved verbatim;
- run ids are pattern-validated before being used as directory names, and
  artifact names may not contain path separators.

Secrets live in `.env` (gitignored; see `.env.example`). Nothing in the pipeline
logs payload content — only run ids, stage names, counts and error codes.

## Running the tests

```bash
.venv/Scripts/python -m pytest
```

**The suite is entirely offline.** No test needs a network, an API key, or a
budget: every model-calling stage runs through its `Fake*Client`, and each
vendor client's own parsing and error-mapping paths are covered by injecting a
stub SDK client. There is no live-API test, by design.

Quality gates:

```bash
.venv/Scripts/python -m ruff format --check . ; .venv/Scripts/python -m ruff check . ; .venv/Scripts/python -m mypy ; .venv/Scripts/python -m pytest
```

## Project layout

```
src/goldpipeline/
    domain/       run ids, error taxonomy — no framework dependencies
    schemas/      Pydantic contracts for every artifact
    prompts/      versioned prompt templates + loader
    services/     normalizer, context builder, market facts, source guard,
                  fencing, integrity, claim resolver, prechecks, content safety,
                  review and finalizer policy, prompt builders, chunking, the
                  run / writer / reviewer / finalizer / publish-gate / publisher
                  stages, and orchestrator.py + run_lock.py above them
    adapters/     source, writer, reviewer, finalizer and publisher protocols;
                  JSON file / Anthropic / OpenAI / Telegram / fake
                  implementations
    storage/      atomic writes, Run directory lifecycle
    config.py     environment settings; the only place a credential lives
    cli.py        argparse entry point
fixtures/         realistic XAUUSD M15 sample data, plus an adversarial message
runs/             generated Runs (gitignored)
tests/
```

The domain and service layers depend on the `AnalysisSource`,
`MarketDataSource`, `WriterClient`, `ReviewerClient` and `FinalizerClient`
protocols, never on a vendor SDK. Only the `adapters/` modules import a vendor
SDK or hold a key, and the two Anthropic stages share their SDK error mapping
through `anthropic_errors.py` so they cannot drift apart in what a failure
means. A real Telegram client or market data provider in a later round drops in
the same way, and nothing downstream changes.

## Scope

**Round 1 — foundation.** Input schemas, validation, normalization, data quality
reporting, immutable Run storage, context building, CLI, fixtures.

**Round 2 — Claude Writer.** Versioned prompt, writer client protocol with
Anthropic and offline implementations, structured output contract, source-price
guard, deterministic price formatting, atomic two-file commit, `write-draft`.

**Round 3 — ChatGPT Reviewer.** Artifact integrity verification, safe claim-path
resolution, deterministic prechecks, versioned reviewer prompt with both
untrusted inputs fenced, reviewer client protocol with OpenAI and offline
implementations, response validation, verdict policy with recorded escalation,
`review-draft`.

**Round 4 — Claude Finalizer.** Verdict routing (passthrough / revise /
block), byte-exact passthrough, finalizer client protocol with Anthropic and
offline implementations, resolution contract with mandatory fixes, deterministic
postchecks and regression comparison, `finalize`.

**Round 5 — final publish gate.** Whole-chain integrity verification, content
safety scanners (instruction-shaped text, credential shapes with redaction,
external factual claims, structure and transport sanity), reuse of the shared
market scanners with a stricter boundary policy, the immutable
`publish_decision.json`, and `gate-publish`. No model, no network, no
credentials.

**Round 6 — Telegram publisher.** Durable publish intent, one immutable attempt
per Run, deterministic chunking with an exact-content invariant, plain-text
transport, bounded 429 retries, and a policy of never retrying an ambiguous
delivery. Publisher client protocol with Telegram and offline implementations,
and `publish`.

**Round 7 — end-to-end orchestrator.** Status-driven stage sequencing over the
existing services, three modes with a non-publishing default, state-aware
resume with terminal publish-side states, lazy client factories, a per-Run
filesystem lock with no automatic stale-lock recovery, a serializable
`PipelineExecutionResult`, and `pipeline-run` / `pipeline-resume`.

**Deliberately not built:** live source adapters (Round 8), schedulers and cron
(Round 9), dashboards, daemons, polling loops, filesystem watchers, automatic
retry or reconciliation of an `UNCERTAIN` attempt, deleting or editing posted
messages, databases, queues, a technical-analysis engine (RSI, MACD, ICT, bias,
trade signals), web or news retrieval, sentiment analysis, and backtesting.

`AnalysisContext` describes facts and carries the raw analysis; it holds no
interpretation. The writer produces prose and an audit trail of the numbers it
used. The reviewer judges that prose and says what to fix. The finalizer fixes
it. The gate decides whether the result may be published, and the publisher
sends it. Round 7 runs that sequence from one command, on demand. Reading fresh
inputs from live sources is Round 8; running it on a schedule is Round 9.
