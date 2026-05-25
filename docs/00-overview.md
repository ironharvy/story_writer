# Definition-phase docs (read before the implementation plan)

These are the "what / why / how" documents that sit upstream of any
implementation plan. They describe the product and its design as it stands
today; they are intentionally concise (≈one page each) and grounded in the
current code.

## Reading order

| # | Doc | Answers |
|---|---|---|
| 01 | [vision.md](01-vision.md) | Why does this exist? What are we building? |
| 02 | [requirements.md](02-requirements.md) | What must it do? (PRD + functional + non-functional) |
| 03 | [architecture.md](03-architecture.md) | How is it structured? (components, data flow, variants) |
| 04 | [algorithms.md](04-algorithms.md) | How does each stage actually work? (algorithms) |
| 05 | [domain-model.md](05-domain-model.md) | What are the entities and terms? |
| 06 | [decisions-and-risks.md](06-decisions-and-risks.md) | Why these choices? What could go wrong? |
| 07 | [acceptance-and-quality.md](07-acceptance-and-quality.md) | How do we know it works? |

## What comes next (the implementation plan)

The roadmap / backlog lives in:

- [pipeline-known-issues.md](pipeline-known-issues.md) — prioritized issue list.
- [handoff-pipeline-fixes.md](handoff-pipeline-fixes.md) — the action plan for it.
- [model-implementation-comparison-2026-05-10.md](model-implementation-comparison-2026-05-10.md) — the model × variant bake-off these docs draw evidence from.

Contributor/coding conventions are in the repo-root [AGENTS.md](../AGENTS.md).

> Status: **pre-implementation reference**. These docs describe the system to
> orient new contributors and frame the backlog; they are not a spec the code
> is formally verified against.
