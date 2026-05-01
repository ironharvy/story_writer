# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

A local-first, CLI-driven DSPy pipeline that produces long-form, coherent, artifact-free story prose from a single user idea. MVP scope is the *quality of the output prose* — nothing else. See [`docs/project-brief.md`](docs/project-brief.md).

## Where to read first

1. [`docs/project-brief.md`](docs/project-brief.md) — what we're building and why.
2. [`docs/decisions.md`](docs/decisions.md) — every design decision with rationale. Append to this rather than silently changing direction.
3. [`docs/architecture.md`](docs/architecture.md) — module graph, Signatures, data model, persistence layout, QA strategy.
4. [`docs/requirements.md`](docs/requirements.md) — testable FRs/NFRs and the shared glossary.
5. [`docs/sprint-backlog.md`](docs/sprint-backlog.md) — epics → tasks.

## Non-negotiables

- **Don't change the shared glossary without a decision entry.** Terms like *premise*, *spine*, *world bible*, *beats*, *embellishment* have exact meanings.
- **The world bible is the single source of truth for facts.** Every later stage must consult it, not its own memory of earlier conversation.
- **DSPy is used as a typed module graph, not a prompt wrapper** ([DEC-0004]). Each stage = one Signature + one Module.
- **No paid services by default.** DeepSeek is gated behind `--allow-paid` ([DEC-0002]).
- **Detection before correction** ([DEC-0007]). QA flags artifacts; auto-revise loops are opt-in per rule.
- **No web UI / auth / images / audio / sharing in MVP** ([DEC-0001]).

## Adding a stage

1. Add a Pydantic output model in `src/story_writer/models/story.py` if needed.
2. Create `src/story_writer/stages/<name>.py` with a `dspy.Signature`, a `StageBase` subclass, and a `_execute()` method.
3. Add the stage to `STAGE_ORDER` in `src/story_writer/stages/__init__.py`.
4. Add a default `ProviderConfig` in `src/story_writer/config.py`.
5. Write a smoke test (mocked Stage if needed) under `tests/`.
6. Append a decision to `docs/decisions.md` if the stage changes the pipeline shape.

## Adding a QA rule

1. Implement `rule_<name>(chapter, ...) -> RuleResult` in `src/story_writer/qa/rules.py`.
2. Add it to `detect_chapter()`.
3. Add positive + negative fixture tests to `tests/test_qa_rules.py`.
4. Update `docs/requirements.md` (FR-020) and the README rule table.

## Code standards

- Python 3.11+. `from __future__ import annotations` everywhere.
- Pydantic v2 for all data models.
- Type hints on every public function.
- `ruff check .` and `ruff format --check .` must pass.
- Tests with `pytest`. New stages need a smoke test; new QA rules need positive + negative cases.
- No comments that just restate the code. Comments only for non-obvious *why*.
- No backwards-compat shims. We're on a clean rebuild branch.

## Environment

- WSL2 / Linux. Local Ollama on `OLLAMA_HOST` (default `http://localhost:11434`).
- Editable install: `pip install -e .[dev]`.
- Run smoke: `pytest -q`.

## What NOT to do

- Don't reach into a stage's internals from the orchestrator. Stages are deep modules with shallow seams.
- Don't merge multiple pipeline stages into one Signature to "simplify". The whole point is per-stage isolation for measurement and DSPy optimization.
- Don't add Langfuse / telemetry / a database / Docker / a web framework. None of those are in MVP scope ([DEC-0010]).
- Don't silently change architecture. Add a decision entry.
