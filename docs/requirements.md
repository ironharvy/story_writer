# Requirements

> References: [DEC-0001], [DEC-0002], [DEC-0003], [DEC-0004], [DEC-0005], [DEC-0006], [DEC-0007], [DEC-0008], [DEC-0009], [DEC-0010]

## Functional Requirements

- **FR-001**: The CLI shall provide `new`, `resume`, `render`, and `inspect` commands.
- **FR-002**: `new` shall accept a raw story idea and create a run directory under `runs/<slug>/`.
- **FR-003**: The clarify stage shall generate 3-7 questions, each with a suggested answer.
- **FR-004**: Interactive mode shall allow the user to accept or edit suggested answers.
- **FR-005**: Non-interactive mode shall accept suggested answers automatically.
- **FR-006**: The premise stage shall produce protagonist, want, obstacle, stakes, genre, length, and summary fields.
- **FR-007**: The spine stage shall produce a six-beat Pixar-style narrative spine.
- **FR-008**: The world bible stage shall produce characters, locations, rules, timeline events, and continuity constraints.
- **FR-009**: The world bible shall be the source of truth for later factual continuity checks.
- **FR-010**: The chapter plan shall size chapter count and include per-chapter beats according to the requested length.
- **FR-011**: The enhancement stage shall produce per-chapter guidance for tension, mystery, theme, setup/payoff, emotional curve, and optional easter eggs.
- **FR-012**: The embellishment stage shall optionally add a small non-plot-changing detail per chapter according to `--embellish-probability`.
- **FR-013**: The prose stage shall write each chapter using the premise, spine, world bible, chapter plan, enhancement notes, and embellishment detail.
- **FR-014**: The orchestrator shall skip completed stages on resume.
- **FR-015**: The renderer shall assemble chapter prose into Markdown or plain text.
- **FR-016**: The inspect command shall print a persisted artifact by stage.
- **FR-017**: Provider routing shall default to Ollama and shall support per-stage model defaults.
- **FR-018**: `--allow-paid` shall be required before any paid provider can be selected.
- **FR-019**: QA shall implement the canonical rule table in this document.
- **FR-020**: `--strict` shall fail a run on hard QA violations.
- **FR-021**: Tests shall be runnable without Ollama by using deterministic fake stages or mocked providers.
- **FR-022**: A release candidate shall not be considered successful while any hard QA finding remains in rendered story output.
- **FR-023**: A run record shall include a customer-acceptance verdict: success, partial, or failed.
- **FR-024**: PM and Tech Lead shall pivot the active backlog when the rendered artifact fails the MVP prose-quality promise.
- **FR-025**: Rendered story output shall include generation parameters, stage model routing, max output tokens, and context-window metadata when known.
- **FR-026**: Unknown or risky context windows shall warn rather than silently proceed.

## Non-Functional Requirements

- **NFR-001**: Python 3.11+.
- **NFR-002**: Pydantic v2 models for persisted artifacts.
- **NFR-003**: Each pipeline stage is one DSPy Signature plus one StageBase subclass.
- **NFR-004**: No database in MVP; filesystem persistence only.
- **NFR-005**: No network dependency is required for unit tests.
- **NFR-006**: `ruff check .`, `ruff format --check .`, and `pytest` are the required validation commands.

## Shared Glossary

- **Idea**: The user's raw starting description.
- **Premise**: The reviewed story foundation: protagonist, want, obstacle, stakes, genre, length, and summary.
- **Spine**: The six-beat Pixar-style structural progression.
- **World bible**: The authoritative fact record for characters, places, rules, timeline, and continuity constraints.
- **Chapter plan**: Ordered chapter summaries with spine alignment and beat lists.
- **Beat**: A discrete scene-level narrative unit inside a chapter.
- **Enhancement guide**: Per-chapter craft guidance for tension, mystery, theme, setup/payoff, emotional curve, and easter eggs.
- **Embellishment**: A small optional detail that enriches prose without changing plot facts.
- **QA rule**: A detection rule that flags possible defects before any correction.

## Constraints

- MVP is CLI only.
- Ollama is the default provider.
- Paid providers are blocked unless explicitly allowed.
- The world bible must be passed into chapter planning, prose, and QA.
- No web framework, telemetry, Langfuse, Docker, auth, or database in MVP.

## CLI Contract

| Command | Options |
|---|---|
| `story-writer new` | `--idea`, `--length short|novella|novel`, `--slug`, `--embellish-probability`, `--non-interactive`, `--allow-paid`, `--strict`, `--skip-qa-embeddings`, `--model`, `--profile fast|quality|tiny` |
| `story-writer resume` | `slug`, `--non-interactive`, `--strict`, `--model`, `--profile fast|quality|tiny` |
| `story-writer render` | `slug`, `--fmt md|txt`, `--out` |
| `story-writer inspect` | `slug`, `--stage` |

`--skip-qa-embeddings` is accepted for compatibility with the CLI contract, but the MVP implementation does not download embedding models by default.

## Canonical QA Rule Table

| Rule | Severity | Description | Strict behavior |
|---|---|---|---|
| R1 | hard | Empty or whitespace-only chapter prose | fail |
| R2 | hard | Meta-framing openers such as "In this chapter" | fail |
| R3 | hard | AI assistant leaks such as "as an AI" | fail |
| R4 | hard | Affirmation openers such as "Certainly!" or "Here is" | fail |
| R5 | hard | Proper nouns not present in the world bible allowlist | fail |
| R6 | soft | Missing expected world-bible anchors in a chapter | warn |
| R7 | soft | Repeated sentences inside a chapter | warn |
| R8 | soft | Repeated sentences across prior chapters | warn |
| R9 | soft | Repeated chapter opening signature across prior chapters | warn |

## Open Questions

- **SPIKE-001**: Whether the installed default routing (`qwen3:latest`, `gemma4:26b`, `gemma3:4b`) should be replaced after quality comparison.
- **SPIKE-002**: Whether DSPy structured output needs extra retry adapters for weaker local models.
- **SPIKE-003**: Whether long-form requests should become hard-blocked, warned, or automatically rerouted when estimated prompt/output budget approaches model context limits.
