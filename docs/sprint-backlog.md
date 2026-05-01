# Sprint Backlog

> References: [DEC-0001], [DEC-0002], [DEC-0003], [DEC-0004], [DEC-0005], [DEC-0006], [DEC-0007], [DEC-0008], [DEC-0009], [DEC-0010]

## Release Status

> References: [DEC-0020]

**Blocked / partial.** The CLI pipeline runs against local Ollama, but the generated story is not customer-acceptable. PM and Tech Lead have pivoted the active iteration from infrastructure completion to prose-quality closure.

## [EPIC-003] Prose Quality Release Blocker

> References: [DEC-0020]

- **[STORY-008] Hard QA findings are resolved before delivery**
  - [TASK-015] Add a bounded revision loop for hard QA failures such as unknown proper nouns.
  - [TASK-016] Add tests proving hard QA failures can be corrected or clearly block release.
  - [TASK-017] Ensure revised prose cannot introduce new world-bible facts without updating the bible through a reviewed stage.

- **[STORY-009] Repeated chapter openings are prevented**
  - [TASK-018] Update prose-stage context and prompt contract to require chapter-specific openings.
  - [TASK-019] Add a regression fixture for repeated first paragraphs across chapters.
  - [TASK-020] Extend QA to flag repeated openings separately from whole-sentence repetition.

- **[STORY-010] PM acceptance gate is explicit**
  - [TASK-021] Add a release checklist requiring a rendered story read-through.
  - [TASK-022] Add a customer-acceptance verdict to run records: success, partial, or failed.
  - [TASK-023] Treat any remaining hard QA finding as release-blocking.

- **[STORY-011] Tech Lead root-cause analysis is completed**
  - [TASK-024] Analyze whether continuity drift comes from world-bible omissions, prose prompt leakage, or missing revision architecture.
  - [TASK-025] Compare `qwen3:latest` and `gemma4:26b` on prose stage quality with the same run inputs.
  - [TASK-026] Document the chosen correction strategy before implementation.

- **[STORY-012] Context limits are visible**
  - [TASK-027] Persist model generation parameters and context metadata per stage.
  - [TASK-028] Render generation parameters in final story output.
  - [TASK-029] Warn on unknown or risky context-window estimates instead of silently proceeding.
  - [TASK-030] Add tests for known model context metadata and unknown/risky context warnings.

## [EPIC-001] CLI Pipeline MVP

- **[STORY-001] User can start a story run from one idea**
  - [TASK-001] Add package scaffold and CLI entry point.
  - [TASK-002] Add run manifest and artifact persistence.
  - [TASK-003] Add orchestrator stage walker with resume behavior.

- **[STORY-002] User can review foundational story artifacts**
  - [TASK-004] Add clarify, premise, spine, and world bible stages.
  - [TASK-005] Add interactive accept/edit/regenerate review helpers.

- **[STORY-003] System can draft chapter prose from structured artifacts**
- [TASK-006] Add chapter plan with embedded beats, enhancement, embellishment, and prose stages.
  - [TASK-007] Ensure every prose stage receives the world bible and chapter plan.

- **[STORY-004] System can detect quality defects**
  - [TASK-008] Add QA models and chapter detection rules.
  - [TASK-009] Add strict-mode failure behavior.

- **[STORY-005] User can inspect and render outputs**
  - [TASK-010] Add `inspect` command.
  - [TASK-011] Add Markdown and text rendering.

## [EPIC-002] Validation

- **[STORY-006] Developers can validate without Ollama**
  - [TASK-012] Add unit tests for models, persistence, render, QA, and orchestrator resume.
  - [TASK-013] Add fake-stage tests for pipeline flow.

- **[STORY-007] Local model smoke can be exercised**
  - [TASK-014] Add an integration test that skips when Ollama is unavailable.

## [SPIKE-001] Local Model Selection

- Compare available Ollama models for structured output and prose quality.

## [SPIKE-002] DSPy Structured Output Reliability

- Evaluate whether local models need parser retries or JSON repair.
