# Story Service — Design Concept

This document is the living theory of what we're building. It is not a spec, contract, or roadmap. It is the shared understanding behind our decisions, updated as our understanding changes.

When you change this document, you are changing the design. When you change something without changing this document, you are creating drift.

## 1. Purpose

A service that generates **interesting, cohesive stories** — including at novel length (War-and-Peace scale, ~500k+ words) — from a user's idea. The analogy is Suno for music or Midjourney for images, but the analogy is loose: stories are consumed over time, not seconds, and they fail differently. A song with weak structure is forgettable; a 500k-word novel with weak structure breaks reader trust by chapter three.

The differentiating problem is therefore not *generating prose* (LLMs do this well) but **maintaining cohesion that scales** — characters that stay themselves, plot threads that pay off, worlds whose rules don't quietly mutate, voices that don't drift. The longer the artifact, the more the system must remember and honor what it already committed to.

## 2. Operating principles

We design from these books and the principles distilled from them:

- *A Philosophy of Software Design* — John Ousterhout
- *The Pragmatic Programmer* — David Thomas & Andrew Hunt
- *The Design of Design* — Frederick P. Brooks
- *Domain-Driven Design* — Eric Evans
- *Extreme Programming Explained* — Kent Beck

Concepts we hold ourselves to:

- **Software entropy.** Software tends toward collapse when changes are made without consideration of the whole. The design concept is what keeps the whole coherent.
- **Code is not cheap.** Bad code is more expensive than ever in the age of code generation — a codebase that is hard to change cannot take advantage of AI's leverage. Quality is leverage.
- **The design concept.** This document. An invisible, ephemeral theory, made temporarily visible here. The asset is the understanding, not the file.
- **Ubiquitous language.** A shared domain vocabulary used identically in code, in this document, in prompts. See §4.
- **Deep modules.** A small interface hiding substantial functionality. We prefer one deep module over a web of shallow modules with complex interfaces.
- **Tactical vs. strategic programming.** AI executes tactically; the human authors the design strategically. This document is the strategic record.

## 3. Phase plan

**Phase 1 — R&D.** Prove the engine works. Deliverable is a **CLI**: takes an Idea (text), produces a directory of artifacts (Premise, Bible, Outline, Manuscript). No users, no auth, no API, no DB, no UI. Inputs and outputs are files. We iterate on quality until we can produce a cohesive novel-length Manuscript that we genuinely want to read.

**Phase 2 — Service.** Wrap the proven engine in a service with accounts, async jobs, email notification, library, reading experience. Free vs paid tier (paid produces novel length). This is conventional platform engineering and should not consume design oxygen until Phase 1 is done.

**Phase 3+ — Extensions.** Audiobook generation, image generation per scene/chapter, deeper Author-involvement modes (stage-gated review, collaborative co-writing).

Until Phase 1 produces stories we believe in, every architectural concern that does not move that needle is noise.

## 4. Ubiquitous language

Use these terms exactly — in code, in this document, in prompts. Adding a synonym (e.g., "book" for "Story") is technical debt; rename or push back.

- **Author** — a user who directs the creation of a Story. Not a typist; a director.
- **Reader** — a user who consumes a published Story.
- **Story** — the domain root. Has a Bible, an Outline, a Manuscript, metadata, lifecycle state.
- **Idea** — the Author's input. A paragraph or so. The seed.
- **Premise** — distilled from the Idea. Structured statement of central conflict, story question, genre, scope.
- **Bible** — the canonical truth about a Story's world, characters, plot, voice, themes. Structured, queryable, exhaustive. Built before prose. Grown append-only.
- **World** — Bible sub-component: setting, era, geography, rules (magic/tech/physics/economics/politics), history.
- **Character** — Bible sub-component: identity, motivation, voice, relationships, arc.
- **Plot Skeleton** — Bible sub-component: major beats, story question, ending, primary and subordinate spines.
- **Spine** — borrowed from Pixar. The dramatic shape of a thread, in the form *"Once upon a time… every day… but one day… because of that… until finally… and ever since."* Used both as generation constraint and as verification probe.
- **Outline** — the structural plan derived from the Bible: acts, chapters, scenes, with per-scene beats (who is present, where, what changes, what is set up, what is paid off). Decides plot; does not decide prose.
- **Scene** — the unit of prose generation. A self-contained dramatic unit, typically a few hundred to a few thousand words. One time, one place, one purpose.
- **Manuscript** — the prose itself. What a Reader reads. Produced scene by scene against Bible + Outline.
- **Extract** — the sealed query interface to the Bible. Given a scene context, returns the minimum sufficient slice of the Bible to write or verify that scene. The only path callers have to Bible content.
- **Consistency Checker** — the post-generation verifier. Reads generated prose, extracts factual claims, queries the Bible via Extract, flags or revises contradictions.

## 5. Bounded contexts

Five contexts. Boundaries enforced. Most design energy goes to the first; the rest are conventional and we try not to invent there.

- **Story Authoring** *(P0, the deep one)* — the pipeline from Idea to Manuscript. Where the AI lives. Where cohesion is enforced.
- **Story Library** *(P1)* — published Stories. Discovery, search, browse. Reader-facing.
- **Reading** *(P1)* — the experience of consuming a Manuscript. Pagination, progress, bookmarks. *(P2 extensions: audiobook output, per-scene/chapter image generation.)*
- **Identity & Accounts** *(P1)* — who is who. Author/Reader roles. Free/paid tier. Billing.
- **Sharing & Publishing** *(P2, low priority)* — the act of moving an Authoring artifact into the Library. Optional later: comments, ratings, follows.

## 6. The Authoring pipeline

A linear pipeline of named, persisted stages. Each stage's output is a structured artifact that the next stage consumes. The pipeline is the deep module of this service.

```
Idea → Premise → Bible → Outline → Manuscript
```

Bible construction is itself a sub-pipeline:

```
Premise → World → Principal Cast → Plot Skeleton → Supporting Cast + Locations → Voice / Tone / Themes
```

Each artifact is persisted. Failures are checkpointed: if scene 412 of 800 fails, we resume, not restart.

In Phase 1 the pipeline runs end-to-end without Author intervention. In later phases the same stages can be gated for Author review and editing — gates are a UI change, not a rewrite. We will not foreclose this by building Phase 1 as a monolithic black box.

## 7. The Bible

The Bible is the deep module. If the Bible is rich and self-consistent, the Manuscript follows almost mechanically. If the Bible is thin or contradictory, no amount of prose-level polish saves us.

Properties:

- **Built before prose.** Exhaustive by the time scene generation starts. Append-only refinements during writing (a new minor character introduced in chapter 12) are allowed; contradictions of established facts are not.
- **Structured, not prose.** Typed entities (Character, Location, WorldRule, Faction, PlotThread, Scene, etc.) with typed relationships. Graph-shaped data model.
- **Queryable, not loadable.** At novel length the Bible is too large to fit alongside a scene in any single LLM call. All access goes through **Extract**.
- **AI-generated from the Idea.** Bible construction is arguably the most important AI task in the system — Bible errors propagate to every scene.

We do not commit to a graph-DB product. The model is graph-shaped; the implementation can stay on a relational store until Extract's query patterns prove otherwise. See §11.

## 8. Decision split

Three separate decision domains, each handled by a different stage:

- **Bible decides world and character.** Who Pierre is. How magic works. What the faction politics are.
- **Outline decides plot.** What happens in chapter 12, scene 3. Who is there. What changes. What is set up. What is paid off.
- **Scene Generator decides prose.** The actual sentences. Dialogue. Description. Pacing within the scene.

This split is what makes scene generation *bounded*: the model is not deciding what happens, it is rendering a pre-decided scene against pre-decided world/character constraints. Cohesion is enforced architecturally — not by hoping a long prompt holds.

## 9. Cohesion enforcement

Two layers, both used:

- **Constrain on generation.** The Scene Generator receives a tight prompt: relevant Bible extract, the Outline beat for this scene, the immediately preceding context. Constraints catch the obvious.
- **Verify after generation.** The Consistency Checker reads the generated prose, extracts factual claims, queries the Bible, and either accepts the scene or sends it back with specific corrections for revision.

The cost (more LLM calls per scene) is accepted. Model and provider selection per stage is a knob we can turn later (cheap models for early stages, premium models for prose, possibly different providers per task); cost concerns should not shape architecture now.

## 10. Operational shape

Phase 2+ runtime model:

- **Async.** Generation is a background job, not a request/response.
- **24-hour soft ceiling** for novel-length runs.
- **Email on completion.** Author submits an Idea, gets a "we're working on it," gets an email when the Manuscript is ready.
- **Checkpointed.** Each stage's output is persisted. Stage failures resume, not restart.
- **Progressable.** Authors can poll a status (which stage is running, how far through scenes).

In Phase 1 this is moot — the runner is a CLI invoked synchronously, and persistence is files on disk.

## 11. Storage strategy

- **Phase 1: filesystem.** One directory per Story. JSON/YAML for structured artifacts. Diff-able, inspectable, schema-mutable for free.
- **Phase 2: relational DB (Postgres) behind a repository interface.** Bible as typed records plus an edges table. The repository interface is the only seam the rest of the system sees.
- **Graph DB: deferred.** The Bible is graph-shaped, but graph-shaped data does not require a graph DB product at our scale. We migrate if and when Extract's query patterns on a relational store demonstrably hurt — not before.

Extract is the *only* module that knows how the Bible is stored. Nothing else queries Bible storage directly. That seam is what keeps storage decisions reversible.
