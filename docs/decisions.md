# Decisions

## DEC-0001: MVP is CLI only

- **Date**: 2026-04-29
- **Phase**: Intake
- **Decided by**: CEO
- **Decision**: Build a CLI-only MVP and defer web service, web UI, auth, images, audio, and sharing.
- **Rationale**: The success metric is prose quality; a web surface would add product and infrastructure work before the generation loop is proven.
- **Alternatives considered**: Start with a web service. Rejected because it adds non-writing concerns to the MVP.
- **Affects**: `docs/project-brief.md`, `docs/requirements.md`, `docs/architecture.md`
- **Status**: Accepted

## DEC-0002: Local Ollama is the default provider

- **Date**: 2026-04-29
- **Phase**: Intake
- **Decided by**: CEO
- **Decision**: Use local Ollama by default and require `--allow-paid` before paid providers are used.
- **Rationale**: The user has Ollama running and wants a local-only MVP.
- **Alternatives considered**: Hosted LLM default. Rejected because it violates local-first expectations and can incur cost.
- **Affects**: `docs/requirements.md`, `docs/architecture.md`, `src/story_writer/config.py`
- **Status**: Accepted

## DEC-0003: Use review gates for foundational artifacts

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: Business Analyst
- **Decision**: Premise, spine, and world bible are presented for review before downstream stages run.
- **Rationale**: Bad foundations compound across long-form generation; review gates keep human intent visible.
- **Alternatives considered**: Fully unattended generation. Rejected for interactive MVP because it hides early story drift.
- **Affects**: `docs/requirements.md`, `src/story_writer/orchestrator.py`, `src/story_writer/interactive.py`
- **Status**: Accepted

## DEC-0004: DSPy is a typed module graph

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: Tech Lead
- **Decision**: Each stage is one DSPy Signature plus one StageBase subclass.
- **Rationale**: Per-stage isolation makes behavior measurable, retryable, and optimizable.
- **Alternatives considered**: One large prompt or one monolithic module. Rejected because it hides failure modes.
- **Affects**: `docs/architecture.md`, `src/story_writer/stages/`
- **Status**: Accepted

## DEC-0005: World bible is the fact source of truth

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: Business Analyst
- **Decision**: Later planning, prose, and QA stages must consult the world bible for story facts.
- **Rationale**: Long-form continuity needs a stable artifact rather than model memory.
- **Alternatives considered**: Let each stage infer facts from all prior text. Rejected because it invites drift.
- **Affects**: `docs/requirements.md`, `docs/architecture.md`, `src/story_writer/models/story.py`
- **Status**: Accepted

## DEC-0006: Persist every stage artifact

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: DevOps
- **Decision**: Persist all stage outputs under `runs/<slug>/` and use a manifest for stage status.
- **Rationale**: Local-first generation needs resumability, inspectability, and debugging artifacts.
- **Alternatives considered**: In-memory execution only. Rejected because long model runs can fail or be interrupted.
- **Affects**: `docs/architecture.md`, `src/story_writer/run_store.py`
- **Status**: Accepted

## DEC-0007: QA detects before correction

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: QA Engineer
- **Decision**: QA rules flag defects; automatic correction loops are out of MVP unless explicitly enabled later.
- **Rationale**: Detection is easier to test and avoids uncontrolled rewrite loops.
- **Alternatives considered**: Auto-revise every flagged chapter. Rejected because it can degrade prose or introduce new facts.
- **Affects**: `docs/requirements.md`, `src/story_writer/qa/rules.py`
- **Status**: Accepted

## DEC-0008: Use filesystem storage instead of a database

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: DevOps
- **Decision**: Use JSON and text files under `runs/` for MVP persistence.
- **Rationale**: The CLI is local-only and single-user; a database would not improve the core prose loop.
- **Alternatives considered**: SQLite. Rejected for MVP to reduce operational surface.
- **Affects**: `docs/architecture.md`, `src/story_writer/run_store.py`
- **Status**: Accepted

## DEC-0009: Embellishment must not change plot facts

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: Business Analyst
- **Decision**: Random embellishments are small details only and cannot alter plot, continuity, or chapter outcomes.
- **Rationale**: The feature should enrich texture without destabilizing the plan.
- **Alternatives considered**: Random plot twists. Rejected because they undermine planning and continuity.
- **Affects**: `docs/requirements.md`, `src/story_writer/stages/embellish.py`
- **Status**: Accepted

## DEC-0010: No telemetry, Docker, or web framework in MVP

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: Tech Lead
- **Decision**: Do not add Langfuse, telemetry, Docker, database, or web framework in the MVP rebuild.
- **Rationale**: These do not directly improve the first-pass prose quality loop.
- **Alternatives considered**: Add deployment scaffolding now. Rejected because deployment is not the MVP target.
- **Affects**: `docs/project-brief.md`, `docs/architecture.md`
- **Status**: Accepted

## DEC-0011: Intake pushback on web-service scope

- **Date**: 2026-04-29
- **Phase**: Intake
- **Decided by**: CEO
- **Decision**: The user mentioned a web service, but MVP delivery remains CLI only.
- **Rationale**: The same user request explicitly allows limiting MVP to CLI, and AGENTS.md marks web UI/auth/sharing out of scope.
- **Alternatives considered**: Build HTTP API first. Rejected until story quality is proven locally.
- **Affects**: `docs/project-brief.md`, `docs/sprint-backlog.md`
- **Status**: Accepted

## DEC-0012: Requirements pushback on glossary drift

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: Business Analyst
- **Decision**: Use the repository terms `premise`, `spine`, `world bible`, `beats`, and `embellishment` rather than introducing parallel names.
- **Rationale**: The project instructions make these terms non-negotiable and downstream models depend on shared meaning.
- **Alternatives considered**: Rename `embellishment` to `random thing`. Rejected because it is less precise.
- **Affects**: `docs/requirements.md`
- **Status**: Accepted

## DEC-0013: Architecture pushback on database and service scaffolding

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: Tech Lead
- **Decision**: Keep persistence in `RunStore` and avoid introducing a service layer for future web use.
- **Rationale**: A premature service layer would produce shallow modules before the CLI contracts are stable.
- **Alternatives considered**: Add FastAPI-compatible service classes now. Rejected as out of MVP scope.
- **Affects**: `docs/architecture.md`, `src/story_writer/run_store.py`
- **Status**: Accepted

## DEC-0014: Architecture review resolved documentation drift before execution

- **Date**: 2026-04-29
- **Phase**: Architecture
- **Decided by**: Tech Lead
- **Decision**: Remove the separate `chapter_beats` stage from the MVP contract, set provisional local routing to `qwen2.5:14b`, document the full CLI contract, and make R1-R8 the canonical QA table.
- **Rationale**: Independent review found drift between README, requirements, and architecture that would otherwise produce mismatched code.
- **Alternatives considered**: Preserve the README's separate `chapter_beats` stage. Rejected because the user described beats as part of chapter planning and the simpler stage graph is enough for MVP.
- **Affects**: `README.md`, `docs/requirements.md`, `docs/architecture.md`, `docs/sprint-backlog.md`, `src/story_writer/config.py`
- **Status**: Accepted

## DEC-0015: Review pushback resolved unsafe run reuse and CLI drift

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: Tech Lead
- **Decision**: Reject duplicate `new` slugs, validate slugs as single safe path segments, validate profile names before orchestration, and lazy-load DSPy-heavy modules outside help/inspect/render startup.
- **Rationale**: Independent review found stale artifact reuse, traversal-capable slugs, raw profile crashes, and noisy local-first CLI startup.
- **Alternatives considered**: Allow overwriting existing run directories. Rejected because it can mix a new idea with old story artifacts.
- **Affects**: `src/story_writer/run_store.py`, `src/story_writer/cli.py`, `src/story_writer/orchestrator.py`
- **Status**: Accepted

## DEC-0016: Review pushback resolved QA false positives and per-chapter records

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: QA Engineer
- **Decision**: Expand R5 common-word handling, add regression coverage for sentence-start pronouns, and record per-chapter embellish/prose manifest keys instead of overwriting root stage records.
- **Rationale**: Strict QA must not fail ordinary prose, and multi-chapter persistence must accurately describe chapter-level work.
- **Alternatives considered**: Disable R5. Rejected because unknown-name detection is useful when calibrated.
- **Affects**: `src/story_writer/qa/rules.py`, `src/story_writer/orchestrator.py`, `tests/test_qa_rules.py`
- **Status**: Accepted

## DEC-0017: Legacy sample artifacts are tolerated but not evidence of prose quality

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: QA Engineer
- **Decision**: Make rendering tolerate legacy chapter JSON shapes, but classify the existing `runs/demo-hollow` prose as a known bad sample because it has a blank chapter and a truncated chapter.
- **Rationale**: The renderer should not crash on existing local artifacts, but broken legacy prose cannot be used as delivery evidence for the MVP quality bar.
- **Alternatives considered**: Rewrite the sample story by hand. Rejected because it would not validate the live model path.
- **Affects**: `src/story_writer/models/story.py`, `src/story_writer/render.py`, `runs/demo-hollow/`
- **Status**: Accepted

## DEC-0018: Align default routing with verified Ollama models

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: DevOps
- **Decision**: Use `qwen3:latest`, `gemma4:26b`, and `gemma3:4b` as default local Ollama routing because those models are installed and reachable in the user's environment.
- **Rationale**: The previous provisional `qwen2.5:14b` default was not present in `ollama list`, which would make the first real generation fail.
- **Alternatives considered**: Ask the user to pull `qwen2.5:14b`. Rejected because suitable local models are already installed.
- **Affects**: `src/story_writer/config.py`, `src/story_writer/orchestrator.py`, `README.md`, `docs/requirements.md`, `docs/architecture.md`
- **Status**: Accepted

## DEC-0019: Live generation findings remain detection-only

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: QA Engineer
- **Decision**: Treat the live smoke story's `Liora` continuity defect and repeated chapter openings as known output-quality issues rather than adding automatic revision in this iteration.
- **Rationale**: DEC-0007 requires detection before correction; the live run proved QA can catch these defects, but correction loops need their own design and tests.
- **Alternatives considered**: Patch the generated story artifact manually. Rejected because it would not validate the pipeline.
- **Affects**: `runs/smoke-ollama-real/`, `docs/daedalus-runs/2026-04-29-cli-story-writer-mvp/validation-summary.md`
- **Status**: Accepted

## DEC-0020: PM and Tech Lead block release and pivot to prose quality

- **Date**: 2026-04-29
- **Phase**: Review & Delivery
- **Decided by**: PM, Tech Lead
- **Decision**: Reclassify the current iteration as partial / not customer-acceptable and block release until generated prose passes the MVP quality bar.
- **Rationale**: The live Ollama run proved the engineering path works, but the rendered story contains a hard continuity defect and visibly repeated chapter openings. That fails the product promise: coherent, interesting, artifact-free story prose.
- **Alternatives considered**: Ship the CLI scaffold and list prose quality as a known issue. Rejected because prose quality is the MVP, not a secondary defect.
- **Affects**: `docs/sprint-backlog.md`, `docs/requirements.md`, `docs/daedalus-runs/2026-04-29-cli-story-writer-mvp/`
- **Status**: Accepted

## DEC-0021: Add customer-acceptance gate to the Daedalus improvement backlog

- **Date**: 2026-04-29
- **Phase**: Postmortem & Skill Learning
- **Decided by**: CEO, PM, Tech Lead, QA
- **Decision**: Record that Daedalus needs an explicit product-acceptance gate owned by PM and QA, plus a Tech Lead root-cause loop when the primary artifact fails.
- **Rationale**: The process initially risked treating a technically executable pipeline as delivery even though the customer-facing artifact failed the MVP promise.
- **Alternatives considered**: Leave the lesson only in conversation. Rejected because the skill-improvement work needs durable evidence.
- **Affects**: `docs/daedalus-runs/2026-04-29-cli-story-writer-mvp/skill-improvement-findings.md`
- **Status**: Accepted

## DEC-0022: Add bounded QA revision and repeated-opening detection

- **Date**: 2026-04-29
- **Phase**: Execution
- **Decided by**: Tech Lead, QA Engineer
- **Decision**: Add R9 repeated-opening detection, strengthen the prose stage contract against invented names and repeated openings, and allow one bounded prose rewrite for hard QA failures before final QA classification.
- **Rationale**: The live smoke story failed customer acceptance because the prose stage invented `Liora` and repeated chapter openings. Detection alone is insufficient for a release candidate, but unbounded auto-revision would violate the MVP's controlled QA approach.
- **Alternatives considered**: Manually edit generated stories. Rejected because it does not improve the pipeline. Add unlimited revision loops. Rejected because it risks unstable generation and hidden drift.
- **Affects**: `src/story_writer/stages/prose.py`, `src/story_writer/orchestrator.py`, `src/story_writer/qa/rules.py`, `docs/requirements.md`, `docs/sprint-backlog.md`
- **Status**: Accepted

## DEC-0023: Add deterministic fallback for persistent unknown names

- **Date**: 2026-04-29
- **Phase**: Execution
- **Decided by**: Tech Lead, QA Engineer
- **Decision**: After bounded model rewrites are exhausted, replace remaining R5 unknown-name spans with generic wording so final QA can enforce the world-bible contract.
- **Rationale**: Live smoke runs showed the prose model repeatedly invented names for an unnamed family member despite QA feedback. A deterministic fallback keeps the world bible authoritative and prevents release-blocking unknown names from persisting silently.
- **Alternatives considered**: Increase rewrite attempts. Rejected because repeated calls still leave the model free to invent another name. Manually edit artifacts. Rejected because it does not improve the pipeline.
- **Affects**: `src/story_writer/orchestrator.py`, `tests/test_orchestrator_resume.py`, `runs/smoke-quality-fix-03/`
- **Status**: Accepted

## DEC-0024: Surface generation parameters and context constraints

- **Date**: 2026-04-29
- **Phase**: Requirements
- **Decided by**: PM, Tech Lead
- **Decision**: Rendered stories include generation parameters, stage routing, token limits, context-window metadata, and context warnings. Context risk is warning-only for now, not a hard block.
- **Rationale**: Model context size is a core constraint for long-form generation. Users need visibility into which models/settings produced a story, and long-story/small-model combinations should not fail silently.
- **Alternatives considered**: Hard-block long stories on small or unknown models. Rejected for now because the pipeline is per-stage/per-chapter and needs more measured evidence before denying a user request.
- **Affects**: `src/story_writer/config.py`, `src/story_writer/models/run.py`, `src/story_writer/orchestrator.py`, `src/story_writer/render.py`, `tests/test_context_limits.py`, `tests/test_render.py`
- **Status**: Accepted
