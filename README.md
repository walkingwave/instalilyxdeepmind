# instalilyxdeepmind

Multi-agent code-repair harness for the [Google DeepMind × InstaLILY Toronto
hackathon](https://toronto26.instalily.ai/) (kickoff Sept 22, submit Sept 29).

The published rubric is *"every task is scored out of 100, weighted to fixes
that survive tests you have not seen."* Everything here is built around that
one sentence: the system is optimized to produce patches that **generalize**,
and the local scoring mirrors the same weighting so improvements are
measurable before the real task exists.

## Setup

```bash
pip install -e .
cp .env.example .env     # then fill in at least one API key
python verify.py         # proves keys, model ids, and tool round-trip work
```

## Commands

```bash
python -m eval.run --task toy --mode none     # grade the untouched repo (no API calls)
python -m eval.run --task toy --mode single   # single-agent baseline
python -m eval.run --task toy --mode multi    # multi-agent orchestration
python -m pytest tests -q                     # harness self-tests, offline
```

Every run writes a scorecard to `eval/results/` and a full JSONL trace to
`tracing/runs/`.

## Scoring

Local scoring (`eval/grade.py`) mirrors the stated rubric:

| Component | Weight | Meaning |
|---|---|---|
| target | 30 | the reported failure now passes |
| regression | 30 | nothing that used to pass broke |
| hidden | 40 | held-out tests, prorated |

`resolved` is the strict bar — all three green.

## The toy task, and why it looks like this

`fixtures/toy_repo` has one planted bug with two independent causes. The
visible failing test exercises only one of them. So:

| Patch | Visible suite | Hidden suite | Score |
|---|---|---|---|
| none | 4/5 | 2/5 | 46 |
| sort only ("lazy fix") | **5/5** | 2/5 | 76 |
| edit the test to match the bug | 5/5 | 2/5 | 76 |
| restore the documented contract | 5/5 | 5/5 | **100** |

The second row is the whole problem: the agent sees a completely green suite
and is still 24 points short. Held-out tests live in `fixtures/toy_hidden/`
and are copied into the workspace **only at grading time**, never during the
run. These numbers are pinned by `tests/test_harness.py`, so a change that
breaks the scoring math fails the suite.

## Layout

```
core/       llm interface + providers, tools, agent loop, budget, config
roles/      role configs over the one agent loop (planner/locator/patcher/
            adversary/verifier/synthesizer)
tasks/      task definitions: toy, swebench, and hackathon (written at kickoff)
eval/       batch runner + scoring + committed scorecards
tracing/    JSONL trace logger + viewer
prompts/    role prompts as editable markdown, not string literals
kb/         knowledge base, one note per concept
fixtures/   toy repo and its held-out tests
```

### Design rules

- **One agent loop.** `core/agent.py` is the only loop. A role is a system
  prompt + tool subset + model tier, never a subclass. That is what makes
  repointing the system at the real task cheap.
- **Tiers, not model names.** Call sites ask for `cheap` or `strong`. Swapping
  provider or downgrading under budget pressure is a config change.
- **Trace from the first call.** A six-agent system is not debuggable
  otherwise, and the trace doubles as the cost ledger and the demo artifact.
- **Baseline before improvement.** No orchestration change ships without a
  before/after scorecard on the same instances.

## Provider notes

Two adapters behind one interface, because the rules are not published until
kickoff and a provider mandate must cost one file:

- **Anthropic** — stateless; the session resends the transcript each turn and
  echoes assistant content back verbatim so thinking blocks round-trip.
- **Gemini** — stateful; the Interactions API chains via
  `previous_interaction_id`, so each turn sends only new input.
  `max_output_tokens` and `thinking_level` go inside `generation_config`
  (verified against google-genai 2.24.0).

## Status

Day 0 complete: primitives, toy task, scoring, traces, self-tests — all
runnable offline. Day 1 adds SWE-bench-Lite and the single-agent baseline;
Day 2 adds the orchestration layer. See the plan for the full schedule.
