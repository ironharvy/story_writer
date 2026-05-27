# Deferred Generators

Variant ideas that were prototyped as their own sibling pipelines on `main`
(parallel `pipeline_<x>.py` / `mymain_<x>.py` files) before the registry
existed. Each is **closed but preserved** — the implementation branch
remains, and the design lives here so it can be re-landed as a registered
generator under `generators/<id>.py` once the protocol is stable
(Phases 6–7 of the registry refactor).

This file is the working ledger; deletion is the retirement signal (per
the Generator Lifecycle rules in `AGENTS.md` once that section lands).

---

## `multi_agent` — TRPG-style ensemble simulation

- **Source:** PR #79 (closed) — branch `claude/multi-agent-story-sim-Jhmhe`.
- **Idea:** an ensemble of agents (Game Master, per-character agents,
  Narrator) collaboratively builds the story through turn-based simulation
  rounds. The GM drives pacing and world state; characters act from their
  own perspective and knowledge (no omniscience); the Narrator renders
  each round into polished prose. Continuity is whatever survives in the
  GM's mutable world state at the end of each round.
- **Key files in the PR:** `story_sim/` package (models, agents, engine),
  `sim_main.py` (CLI entry), `test_sim.py` (28 tests covering models,
  helpers, and integration).
- **Why deferred:** shaped as a sibling pipeline, not a registry plug-in.
  Re-land as `generators/multi_agent.py` consuming `DraftingInput` and
  returning `DraftingOutput`; map the GM's world-state carry into
  `DraftingOutput.continuity_artifact`.
- **Re-land checklist:**
  1. Copy `story_sim/` files into `generators/multi_agent/` (or fold into
     a single module if it fits).
  2. Add `@register(id="multi_agent", status="experimental", ...)` and a
     `draft(self, inp: DraftingInput) -> DraftingOutput` adapter that
     wraps the existing engine.
  3. Port `test_sim.py` tests under `tests/generators/test_multi_agent.py`.
  4. Add at least one `bench/fixtures/` idea that exercises the multi-POV
     niche (ensemble, conflicting agendas).
  5. Run `python bench/run.py --generators baseline,multi_agent` and
     confirm `multi_agent` finishes without errors on at least one fixture.

---

## `recursive` — story applied to itself

- **Source:** PR #100 (closed) — branch `claude/recursive-story-generation-YYtg0`.
- **Idea:** the story-generation procedure is applied to itself. Each
  chapter's beats become a child "idea"; the world bible is projected
  down to what the unit actually uses (`ProjectWorldBible`); the parent's
  act-sliced spine is reused; `ExpandUnit` decomposes the unit into
  sub-beats — recurse until a depth cap (default 2: story → chapter →
  scene) or an atomic unit, at which point prose is drafted at the
  leaves and concatenated bottom-up.
- **Key files in the PR:** `story_recursive.py` (`RecursiveWriteStory`
  module + `StoryNode` tree + `ExpandUnit` / `ProjectWorldBible`
  signatures), `pipeline_recursive.py`, `mymain_recursive.py` (variant
  pipeline + CLI), `test_story_recursive.py` (StoryNode assembly +
  leaf-count coverage).
- **Why deferred:** shaped as a sibling pipeline (uses the suffix-coded
  `pipeline_recursive.py` / `mymain_recursive.py` shape we're moving
  away from). `forward` is already pure — adaptation should be small.
- **Re-land checklist:**
  1. Move `story_recursive.py` to `generators/recursive.py`; keep
     `StoryNode` + `ExpandUnit` + `ProjectWorldBible` in the same module
     (cohesion).
  2. Add `@register(id="recursive", status="experimental", ...)` and an
     adapter that runs `RecursiveWriteStory()(...)` over
     `DraftingInput`, then maps the assembled leaves into
     `DraftingOutput.chapters`. `continuity_artifact=None` (no carry).
  3. Surface `--max-depth` (PR #100 added it as a global flag) as a
     generator-specific config: keep `DraftingInput` clean and read
     `max_depth` from a per-generator config block instead of polluting
     the unified `main.py --max-depth` argument.
  4. Port `test_story_recursive.py` under `tests/generators/test_recursive.py`.
  5. Add a `bench/fixtures/` idea that exercises long-arc cohesion (the
     niche the recursion claims to solve via the act-sliced spine).

---

## `editor_in_chief` — pre/post production review wrap

- **Source:** PR #86 (still open as of writing — `.claude/skills/write-story.md`
  edit). Not strictly a generator; an *evaluation/revision wrap* around
  the chapter-drafting step.
- **Idea:** two editor stages bookend chapter writing. Pre-production
  critiques the plan before any prose is committed; manuscript review
  critiques the prose after. Plus per-stage and per-chapter checkpointing
  to `.tmp/story_state.json` so progress survives interruptions.
- **Status:** when the orchestrator (`main.py`) lands, the editor wrap
  fits cleanly as a pre/post hook around `generator.draft(...)`. The
  checkpointing belongs in the orchestrator regardless. Track in this
  file so the design isn't lost; integration is independent of the
  registry protocol.

---

## How to add a new deferred idea

Append a section above with the same shape: source PR / branch, the
idea in one paragraph, key files, why deferred, re-land checklist.
Closing PRs that we want to keep as future variant candidates is
better than leaving them open and stale.
