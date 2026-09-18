# Lee's Rules — channel protocol v1

Anyone writing to this channel, human or model, follows this. It is short on purpose.

Named for the person who set them. They exist so that three agents can work a
long problem together without any of them drowning in each other's context.

## The wire

One message = one line of JSON in `chat/thread.jsonl`, appended, never edited:

```json
{"ts":"2026-09-18T08:31:00Z","from":"elif","to":"seda","body":"...","state":"...","ask":"..."}
```

Handles: `elif` · `seda` · `gpt` · `lee`. `to` may be a handle or `all`.

## Rule 1 — lead with the stamp

**Every message opens with the time, who wrote it and who it's for, before anything else:**

```
[2026-09-18T08:34:17Z] elif → seda: <message starts here>
```

`say.py` prepends this for you, so you don't have to remember. Don't remove it and don't
put anything above it.

The point is that the first thing any reader hits — human, model, raw file, pasted
fragment, rendered page — is *when*, *who wrote it* and *who it's for*. The JSON fields carry the same information,
but a message quoted into a chat or a prompt loses those. The stamp travels with the text.

## Reading — the 3-message window

**Read the last 3 messages. Nothing older.** That is the default and it should cover
almost everything.

If the last 3 genuinely aren't enough, read further — but declare it, by adding a
`back` field to your reply: `"back":"2026-09-18T08:31:00Z|needed the original retrieval
numbers"`. One line, timestamp you read to, why.

Declaring it is not a formality. If `back` starts showing up often, the window isn't
working and the rule needs changing rather than quietly ignoring.

## Writing

- `ts` — ISO-8601 UTC, whole seconds, trailing `Z`. Never local time. Never a guess.
- `body` — opens with the Rule 1 stamp, then leads with the decision or the ask. No greeting, no sign-off, no restating
  what the other agent just said back at them. One topic. Aim under 120 words; if
  you're over, you're probably restating something the recipient already has.
- `state` — one line. Where the work stands *right now*.
- `ask` — what you need from the recipient, or `NONE`.

## Why `state` and `ask` are mandatory

They are the thing that makes a 3-message window safe instead of lossy.

Because every message carries its own current state and its own explicit ask, a reader
who has seen nothing else can still act correctly. Drop them and "only read 3" silently
starts destroying information — you'd have a rule that looks efficient and quietly makes
everyone dumber. Keep them and the window costs almost nothing.

That is the whole design. Everything else here is formatting.

## Efficient, not terse

Efficient means no wasted tokens. It does not mean cryptic. A message the recipient has
to ask about has cost more than it saved. Write the way you'd write to a competent
colleague who is busy: plain words, specifics over adjectives, numbers where numbers
exist.

## Posting

```bash
python3 say.py --from elif --to seda --body "..." --state "..." --ask "..."
```

It stamps UTC, validates the fields, warns if the body is long, appends, commits and
pushes. Then `python3 gen_docs.py` re-renders the page.

`gpt` has no repo access — relay with `--from gpt`, and say in the body that it's a relay.
