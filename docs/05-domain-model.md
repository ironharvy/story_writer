# 05 · Domain Model & Glossary

## Entities

| Entity | Defined in | Fields | Notes |
|---|---|---|---|
| `WorldBible` | `story.py` | `story_title`, `rules_of_the_world: list[str]`, `characters: list[str]`, `locations: list[str]`, `timeline: list[str]` | Static canon produced in the foundation phase. Characters/locations are enhanced free-text strings, not structured records. |
| `PlanEntry` | `story.py` | `chapter_title`, `chapter_beats` | One entry per planned chapter. |
| `Character` | `story.py` | `name`, `description`, `role`, `relationships: list[str]`, `arc` | A structured character dataclass; the live `WorldBible.characters` currently holds strings rather than these. |
| `WorldState` | `world_state.py` | `story_clock`, `characters: list[str]`, `locations: list[str]`, `plot_threads: list[str]`, `key_objects: list[str]`, `recent_events: list[str]` | Variant B only. The *mutable* counterpart to the static bible; evolves per chapter. |
| `Finding` | `qa.py` | `check`, `severity` (`info`/`warn`/`fail`), `message` | One QA observation. |
| `Replacement` | `story_linter.py` | `find`, `replace`, `reason` | One verbatim prose edit proposed by the linter. |
| `ChapterPOV` | `pov_check.py` | `chapter`, `pov` (`first_person`/`third_person`/`mixed`/`other`), `note` | One chapter's classified narration POV. |
| `DSPyConfig` | `dspy_runtime.py` | model / api_key / max_tokens / num_ctx / cache flags / cache_dir | Runtime LM configuration. |

## Relationships

```
Idea ─┬─▶ CorePremise ─▶ Spine ─▶ WorldBible ─▶ [PlanEntry]
      │                                              │
      │                                  ┌───────────┴───────────┐
      └────────────────────────────────▶│ Chapter prose (×N)    │
                                         └───────────┬───────────┘
                          WorldState (B) evolves alongside chapters
                          Finding / ChapterPOV / Replacement describe finished prose
```

## Glossary

- **Idea** — the one-line input prompt; refined into an *updated idea* by the
  clarify step.
- **Core premise** — the story's central dramatic engine in a few sentences.
- **Spine** — the narrative arc as the **Pixar 7-step** beats (once upon a
  time → every day → until one day → because of that → and because of that →
  until finally → ever since that day).
- **World bible** — the *static* canon: rules, locations, timeline, characters.
  Reference material for drafting, not to be quoted verbatim.
- **World state** — the *mutable* snapshot of where the story stands now
  (variant B): clock, character/location state, open threads, objects, recent
  events.
- **Chapter plan** — ordered list of `{title, beats}` per chapter.
- **Story so far** — variant A's rolling plain-text summary threaded between
  chapter drafts.
- **Act hint / spine slice** — the 3-act mapping for a chapter and the spine
  beats it is allowed to see (see [04-algorithms.md](04-algorithms.md)).
- **Variant A / B / C** — baseline / world-state / module drafting strategies.
- **Finding** — a single QA result with a severity.
- **Drift** — a name/POV that diverges from the established canon across
  chapters (name drift = spelling/form; POV drift = narration person).
- **Artifact** — the incrementally-written output markdown file.
