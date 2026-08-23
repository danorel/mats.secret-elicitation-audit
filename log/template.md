<!--
Template for a session log entry. Copy this file to log/YYYY-MM-DD-session-NN.md and fill
it in as you go (or right after the session). Keep section headers and order as-is so every
session log reads the same way - that consistency is the point. Delete a section only if it
is genuinely empty (e.g. no bugs were hit); do not leave "N/A" placeholders.

Style:
- Write findings and issues as: claim first, then the evidence, then what it means for later
  work. A reader should get the headline from the first sentence of each item.
- Bold the one or two numbers/phrases that matter most in a paragraph - not everything.
- Prefer tables over prose for anything with 3+ comparable numbers.
- Link to commit hashes / file paths instead of re-explaining what the code does.
- This file itself (template.md) is not a session log - do not add an entry for it in Done.
-->

# Session N — YYYY-MM-DD

## Summary

<1-3 sentences. The headline: what changed, what was found, what's the bottom line. Someone
reading only this paragraph across all session logs should be able to reconstruct the
project's trajectory.>

## Session info

- **Wall clock:** start–end (timezone), total duration
- **Active compute:** time actually spent on GPU, if different from wall clock (setup,
  waiting, debugging often dominate)
- **Machine:** e.g. rented A100 40GB (Sesterce/Shadeform), or "local, no GPU"
- **Time breakdown:** wall-clock minutes/hours split by what it went to. The three
  buckets:
  - **Code:** writing/debugging scripts, prompts, hooks — anything before a GPU run
    starts
  - **Hypothesis testing:** running experiments (apply + audit, pilots, sweeps) and
    analyzing their output — the GPU-bound and results-reading time
  - **Training:** should be **0h every session** — CLAUDE.md rules out training any
    model; keep the row anyway so a nonzero value is immediately visible as a scope
    violation, not silently dropped
  - These three should roughly sum to the session's wall clock (idle/wait time like the
    HF gate or a long unattended run can be called out separately if it's large)
- **Budget used so far:** cumulative hours against the ~20h total (see CLAUDE.md)

## Goal coming in

<What this session set out to do - normally copied from the previous session's "Next
session" list. If the session's actual focus diverged from that plan, say so and why.>

## Done

<Terse bullet list of completed work items - one line each where possible. Save the
detail for Findings/Results/Issues below; this section is a scannable checklist.>

- ...

## Findings

<One subsection per finding. A finding is anything learned that wasn't already known -
a pattern in the data, a mechanistic explanation, a "why does X happen" answer. Not the
same as a Result (a number) or an Issue (a bug) - a finding is the *interpretation*.>

### <Finding title, phrased as the claim itself>

<Evidence, then implication for later work.>

## Results

<Numbers - reproduction tables, method comparisons, accuracy tables. One table per
comparison, with a one-line caption above stating what's being compared and against what
target/baseline. Always include the control rows alongside the real result, never just
the headline number.>

## Issues hit

<One subsection per bug/problem, in the order encountered. Skip this section entirely if
none were hit.>

### <what broke, in a few words>

- **Symptom:** what you observed
- **Root cause:** why it happened
- **Fix:** what changed (commit hash if committed)

## Decisions

<Choices made this session and why - especially anything where a person picked between
options you presented. Skip if the session was pure execution with no branch points.>

- ...

## Next session

<Concrete, ordered punch list for picking the work back up. Should be specific enough
that a fresh read of just this section (plus the Summary) is enough to resume without
re-reading the whole log.>

1. ...
