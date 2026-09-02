# Producer contract — remote event intake

What an external analysis producer must implement in order to feed
GoldAnalysisPipeline. Written so the producer can be built independently: it
needs nothing from this repository beyond this page.

The producer runs on its own machine. **This pipeline connects out to it; it
never connects in.** There is no inbound port, no receiver and no daemon on the
pipeline side — a scheduled task that already runs every minute does the fetch
as one bounded step.

---

## 1. The endpoint you provide

```
GET  https://<your-host>/outbox/pending?limit=<N>
```

| | |
|---|---|
| Transport | HTTPS only. The pipeline refuses to send a bearer token over plain HTTP |
| Auth | `Authorization: Bearer <ingest token>` |
| Accept | `application/json` |
| Redirects | **Not followed.** A `3xx` is treated as a failed fetch, because following it would re-send the Authorization header to another origin |
| `limit` | Integer chosen by the pipeline. Never return more than asked — a longer list is refused whole |

### Response — HTTP 200

```json
{
  "events": [
    {
      "schema_version": "1",
      "source": "gold_analysis_bot",
      "event_id": "gold_analysis_20260902T151838Z",
      "created_at": "2026-09-02T15:18:38.000000Z",
      "raw_text": "Phân tích XAUUSD: ..."
    }
  ]
}
```

`events` may be empty. That is a normal, successful answer and the common case.

### Other statuses

| Status | Pipeline behaviour |
|---|---|
| `200` | Envelope checked, events admitted individually |
| `401`, `403` | Treated as a configuration fault. Recorded once, not hammered |
| `408`, `429`, `5xx` | Transient. Retried on a later tick |
| `3xx` | Refused (see redirects above) |
| anything else | Refused as an unusable answer |

---

## 2. The event schema

Version `1`. Additional keys are **rejected** — the schema is the list of things
a producer is allowed to influence, and there is deliberately no field that can
select a model, a destination chat, a filesystem path or a publish action.

**Required**

| Field | Rule |
|---|---|
| `schema_version` | Exactly `"1"`. An unknown version is refused, never guessed |
| `source` | Non-empty label for the producer, e.g. `gold_analysis_bot` |
| `event_id` | `^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$` — 8–128 characters. No `:` (invalid in a Windows filename), no leading dot, nothing that could climb out of a directory |
| `created_at` | UTC ISO-8601, `Z` suffix |
| `raw_text` | The analysis, verbatim. Non-empty, at most 200,000 characters |

**Optional** — carried through as data only: `message_date`, `chat_id`
(int or string), `message_id` (int), `author` (string), `metadata` (object).

`raw_text` is treated as **untrusted** for its whole life. It reaches the writer
inside a fence, and instructions written into it are handled as data, not as
commands.

---

## 3. Delivery semantics — read this part

**At-least-once is not merely tolerated; it is the design.**

Keep each event in your outbox and return it on every poll until it ages out.
You do not need a consumer cursor, an acknowledgement, a delivery receipt, or
any record of what this pipeline has taken. There is no ack endpoint, on
purpose: an ack protocol is a second thing that can fail, and it would buy
nothing here.

Duplicates are safe. The pipeline decides admission locally, keyed on
`event_id` plus the SHA-256 of the exact payload bytes:

| Situation | Result |
|---|---|
| New `event_id` | Admitted once. A Run follows |
| Same `event_id`, same payload, offered again | Skipped quietly, no second Run — forever, however many times it is offered |
| Same `event_id`, **different** payload | **Conflict.** Refused, recorded for a human, and nothing already stored is overwritten |
| Fails schema validation | Refused and counted. Valid events in the same response are unaffected |

So: **never reuse an `event_id` for different content.** That is the one rule
whose violation needs a person to resolve. If an analysis is revised, give it a
new id.

**A successful `GET` does not mean a Run was created.** It means bytes were
transferred. The event may be a duplicate, may fail validation, may arrive too
late to be worth pairing with market data, or may be refused by a later stage.
Do not infer pipeline outcomes from HTTP status.

### Ordering and retention

- Return **oldest first**.
- Retain each event for **at least 24 hours**. The pipeline may be offline; a
  retention window is what lets it catch up without you tracking its state.
- Events older than the configured analysis-event age (default **60 minutes**)
  expire without producing an article. This is deliberate: a note about this
  morning's candles should not be published against this evening's market.
  Retention exists to survive an outage, not to publish stale commentary.

### Pagination

Not required today, and `limit` alone is sufficient at this volume. If it ever
becomes necessary it must be an **opaque, server-issued cursor** — never a
client wall-clock timestamp, which silently drops events written during a clock
skew or a slow write.

---

## 4. Credentials

The producer holds exactly **one** secret: the ingest token, and it authorises
reading the outbox and nothing else.

The producer never receives, and must never ask for, the pipeline's Anthropic
key, Telegram bot token, MT5 credentials, configuration file or any filesystem
path. Nothing about the pipeline's internals is shared — including whether a
given analysis was ever published.

Rotating the token is a two-sided change: update the credential store on the
pipeline machine and the producer's copy. A rotated-out token produces `401`,
which is recorded as a configuration fault rather than retried in a loop.

---

## 5. Failure expectations

The pipeline treats this source as strictly optional. If the producer is down,
slow, unauthorised or answering nonsense, the pipeline records a safe code and
carries on with its local work unchanged. There is no alerting obligation on
the producer, and no need to guarantee availability.

Conversely, the producer should assume nothing about how quickly an event is
picked up beyond "typically within a minute or two of appearing in the outbox".

---

## 6. Enabling the pipeline side

Off by default. On the pipeline machine:

```
GOLDPIPELINE_INGEST_ENABLED           = true
GOLDPIPELINE_INGEST_URL               = https://<your-host>
GOLDPIPELINE_INGEST_MAX_EVENTS        = 25       (optional, ≤ 200)
GOLDPIPELINE_INGEST_TIMEOUT_SECONDS   = 10       (optional, ≤ 60)
```

plus the token in the OS credential store:

```
python -m goldpipeline secrets-set INGEST_TOKEN
```

> **Migration note.** These keys are read by name and are *not* members of
> `ConfigKey`, so under `STRICT_PERSISTENT` they are simply absent and the
> feature stays off. Turning it on in production is a deliberate two-step
> change — add the members to `ConfigKey`, then rewrite the authoritative
> config file to match — because `REQUIRED_PRODUCTION_KEYS` is derived from
> that enum and a new key becomes mandatory the moment it ships. Doing only the
> first step would fail the next scheduled tick closed. See the note in
> `schemas/runtime_config.py`.
