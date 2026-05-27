# Benchmark harness

This is the goal-function machinery for the generator registry. It runs
every registered drafting strategy against a set of fixture ideas and
emits a comparison table so we can tell which variants are actually
producing better stories — not just feel like they are.

The objective function (Tier-1 deterministic gates + Tier-2 LLM-judge
checks + Tier-3 qualitative rubric) lives in [`eval-spec.md`](eval-spec.md).

## Pieces

| File | Role |
|---|---|
| `eval-spec.md` | The spec — what "good" means. Source of truth. |
| `rubric.md` | Tier-3 qualitative axes for the LLM judge. |
| `criteria.py` | Tier-1 deterministic runner (wraps `qa.py`); emits a JSON `Scorecard`. |
| `fixtures/` | Idea fixtures: one JSON per (title, idea, chapter count, niche). |
| `run.py` | Drives `(fixture × strategy)` end-to-end against your configured LLM. |
| `score.py` | Walks a run directory, applies `criteria.py`, writes `results.md`. |

Tier-2 / Tier-3 LLM-judge implementations are stubs — the harness today
scores deterministic gates only. The Tier-1 budgets (chapter length,
character presence, name drift, phrase reuse) catch the most common
failure modes well enough to start ranking. The judge wiring is the
next obvious extension.

## Quickstart

```bash
# Run all promoted generators on all fixtures against local Ollama.
python -m bench.run --model qwen3:latest --provider ollama

# Score the results.
python -m bench.score .tmp/bench/<run-id>
```

`bench.run` writes one story per `(fixture, strategy)` under
`.tmp/bench/<run-id>/<fixture>/<strategy>/story.md` and `bench.score`
writes a `scorecard.json` next to each plus a `results.md` summary
at the run root.

## Narrowing what you run

```bash
# A single generator on a single fixture.
python -m bench.run --generators baseline --fixtures 01_thriller

# Multiple generators, all fixtures.
python -m bench.run --generators baseline,world_state

# A specific run id (otherwise UTC timestamp).
python -m bench.run --run-id smoke-1
```

## Adding a fixture

Drop `bench/fixtures/<NN>_<slug>.json`:

```json
{
  "id": "04_unreliable_narrator",
  "title": "The Witness",
  "idea": "...",
  "number_of_chapters": 6,
  "niche": "One sentence: what makes this fixture a useful test."
}
```

Fixtures should each exercise a distinct niche so the comparison
surfaces a *reason* one variant wins, not just a noise margin. The
`niche` field is for the reader; the harness doesn't parse it.

## Scoring a single story

Outside of a full bench run, you can score one story directly:

```bash
python -m bench.criteria .tmp/story.md
```

This is what `bench/score.py` does for each result, just one file at
a time.

## Retirement policy

Generators registered as `experimental` are included in benchmark
sweeps. They must clear the goal function on at least one fixture
within ~4 weeks or auto-deprecate. `promoted` generators are the
default candidates; a new variant beats them on ≥ 50% of fixtures, or
fills a documented niche, to earn promotion. `deprecated` ones get
deleted after one cycle. See AGENTS.md → "Generator Lifecycle".
