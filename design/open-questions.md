# Open Questions

Things we have discussed but not decided, or things we have deferred. Each entry is a thread to pick up later. When we resolve one, move the decision into `concept.md` and remove or condense the entry here.

## Authoring pipeline

- **How exactly the Pixar spine integrates.** Agreed it is a useful primitive both for generation (forces dramatic shape) and verification (a story whose spine is incoherent is failing). At novel scale we likely need a primary spine plus subordinate spines per major character/thread, woven together — but the mechanism for weaving multiple spines is undefined.
- **What the Outline format looks like in detail.** We have said it decides plot at the scene level (who is present, where, what changes, what is set up, what is paid off). The actual schema is undefined.
- **How Extract works mechanically.** Its input contract, its output, how it composes Bible content for a given scene. Possibly the most interesting AI task in the system; not yet designed.
- **How the Consistency Checker works mechanically.** Claim-extraction approach, query patterns against the Bible, revision-loop semantics, accept/reject thresholds.
- **Whether Bible-construction sub-stages can run in parallel** or whether each strictly depends on the prior. Deferred until we have a first running version.
- **What counts as "interesting"** as distinct from "cohesive," and how (if at all) we measure or enforce it. Cohesion has a tractable definition; interestingness is open.

## Author involvement

- **Stage gates for future Author involvement.** In Phase 1 the Author submits an Idea and walks away. In later phases we want the Author able to intervene at every stage. The mechanism — review-and-edit per artifact, freeform comments, targeted corrections, something else — is open.

## Productization

- **Free vs paid tier mechanics.** Probably a word-count cap on free, novel length unlocked on paid. Exact policy deferred to Phase 2.
- **Generation time budget.** 24-hour soft ceiling agreed; the hard target (and how we hit it) is open.
- **Model selection per stage.** Cheap models for some stages, premium for prose. Specific assignments deferred.
- **What "publishing" actually means.** Workflow for moving Author artifact → Library entry. Versioning of published Manuscripts. Whether unpublished Stories are private, link-shareable, or invisible.

## Extensions (Phase 3+)

- **Audiobook generation.** Likely per-scene or per-chapter, with consistent narrator/voice across the Manuscript. Deferred.
- **Image generation per scene/chapter.** Style consistency across hundreds of images is its own cohesion problem. Deferred.

## Storage / infra

- **When to migrate from filesystem to Postgres.** Trigger condition is "Extract's query patterns hurt on filesystem" — what "hurt" means in practice is open.
- **Whether a graph DB ever earns its keep.** Re-evaluate only with concrete evidence of multi-hop queries that do not fit relational comfortably.

## Prior work

- **What to mine from `archive/pre-redesign/`.** The prior implementation contains a story linter, a POV checker, a QA module, and a world-state tracker — direct precursors to our Consistency Checker. We should review them for hard-won lessons before reinventing.
