# Bench results

Committed scorecards from real benchmark sweeps. Each file is a
`<utc-timestamp>.md` produced by `bench/score.py` (the `results.md` summary,
renamed to its run timestamp) so the goal-function history is tracked in git.

## Status

**No sweep has been run yet.** Phase F4 of the Plan A follow-ups requires a
working local Ollama (the harness drives a real model end-to-end — it does not
mock the LLM), and none was available in the environment where F1–F3 landed
(no `ollama` CLI, nothing serving on `:11434`). The deterministic Tier-1
harness and the new Tier-2/Tier-3 judge (F3) are exercised by the unit tests;
what's deferred here is a *live* multi-fixture, multi-generator run.

## How to produce the first sweep

On a machine with Ollama serving `qwen3:latest`:

```bash
# 1. Draft every promoted generator on every fixture (one full pipeline run
#    per (fixture, strategy) pair — minutes to hours depending on the model).
python -m bench.run \
  --generators baseline,world_state,dspy_module \
  --fixtures all \
  --model qwen3:latest \
  --max-tokens 8192

# 2. Score with the LLM judge (Tier-1 + Tier-2/Tier-3).
python -m bench.score .tmp/bench/<run-id> --llm-judge --judge-model qwen3:latest

# 3. Commit the summary under this directory, named by the run's UTC timestamp.
cp .tmp/bench/<run-id>/results.md bench/results/$(date -u +%Y%m%dT%H%M%SZ).md
```

> `--fixtures all` is the default (omitting `--fixtures` runs every fixture
> under `bench/fixtures/`); it is spelled out above to match the F4 command.

## After a sweep: status transitions

Once a real scorecard exists, read it against the Generator Lifecycle policy
in `AGENTS.md`:

- A `promoted` variant that another promoted variant **strictly beats** on the
  goal function (ships on more fixtures, no niche of its own) is a demotion
  candidate.
- Open a tracking issue per proposed transition (`mcp__github__issue_write`)
  with the scorecard evidence.
- **Do not** edit `status="..."` in `generators/<id>.py` without explicit
  human sign-off — that is the real demotion, and the human picks.

No transitions are proposed yet because there is no scorecard to justify one.
