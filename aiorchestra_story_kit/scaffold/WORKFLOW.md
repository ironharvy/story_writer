# Workflow — how this story repo is built

This repo is driven by [aiorchestra](https://github.com/ironharvy/AIOrchestra):
every writing phase is a GitHub issue, the agent opens a PR with exactly one
artifact, an editorial review pass runs, you merge it, then the next phase's
issue is filed. The phase chain mirrors the `story_writer` `write-story` skill.

## The phase chain

| # | Issue title | Reads | Produces (one artifact) | On finish, files |
|---|---|---|---|---|
| 1 | `Idea: <story>` | `story/idea.md` (seed), `README.md` | `story/idea.md` — idea + the 5 interrogative questions, each with a proposed answer and an accepted answer; also appends a `## Style` section to `AGENTS.md` / `CLAUDE.md` | `Premise: <story>` |
| 2 | `Premise: <story>` | `story/idea.md` | `story/premise.md` — the Core Premise paragraph (central conflict, protagonist goal/motivation, stakes, setting, tone, theme) | `Spine: <story>` |
| 3 | `Spine: <story>` | `story/premise.md` | `story/spine.md` — the six-beat template (Once upon a time… / Every day… / One day… / Because of that… / Because of that… / Until finally…) | `World Bible — Rules: <story>` |
| 4 | `World Bible — Rules: <story>` | `story/premise.md`, `story/spine.md` | `story/world/rules.md` — rules/systems of the world (magic, tech, science, law, etiquette, societal norms, exploitable loopholes) | `World Bible — Characters: <story>` |
| 5 | `World Bible — Characters: <story>` | the above + `story/world/rules.md` | `story/world/characters.md` — every significant character (full name, physical description, relationships, role, aspirations, flaws; majors detailed, minors brief) | `World Bible — Locations: <story>` |
| 6 | `World Bible — Locations: <story>` | the above + `story/world/characters.md` | `story/world/locations.md` — every significant place (description, climate, who's there, geography, atmosphere, plot significance) | `World Bible — Timeline: <story>` |
| 7 | `World Bible — Timeline: <story>` | all of `story/world/` | `story/world/timeline.md` — chronological major events, backstory through ending | `Chapter Plan: <story>` |
| 8 | `Chapter Plan: <story>` | `story/premise.md`, `story/world/` | `story/chapter_plan.md` — 3 acts (Setup / Confrontation / Resolution), 3-5 chapters each (~9-15 total), one sentence per chapter, numbered | `Enhancers: <story>` |
| 9 | `Enhancers: <story>` | `story/chapter_plan.md`, `story/world/` | `story/enhancers.md` — per-chapter assignment of: Tension, Mystery (plant/reveal), Theme alignment, Setup/Payoff tracker, Emotional curve, Twist generator, Easter-egg injector | `Chapters: <story>` |
| 10 | `Chapters: <story>` *(or `Chapter NN: <chapter title>` per chapter for long works)* | everything in `story/` | `story/chapters/NN-slug.md` per chapter | `Assemble & Publish: <story>` |
| 11 | `Assemble & Publish: <story>` | all chapters + foundation files | `story/final.md` (sections: Core Premise / Spine / World Bible / Chapter Plan / Enhancers / Final Story), then publishes it | — (terminal) |
| — | `Revise — <path>: <note>` | the named file + its prerequisites | the named file | — (does not advance the chain) |

`Chapters` is one issue for short stories. For novel-length work (e.g.
`story/idea.md` calls for a long book) the agent instead files one `Chapter NN:
<title>` issue per chapter in `story/chapter_plan.md`, in order — each advancing
to the next, the last advancing to `Assemble & Publish`.

## How a phase runs

1. aiorchestra's `discover` stage finds the issue (it carries the `aiorchestra`
   label, plus `claude` so it routes to the Claude Code agent).
2. `prepare` makes a branch `claude/<issue-number>`.
3. `implement` invokes the agent with `.aiorchestra/templates/implement.md`,
   which routes on the issue-title prefix to exactly one phase. The agent reads
   every file under `story/`, writes the one artifact, updates `story/STATUS.md`,
   and (unless this is the terminal phase) files the next issue.
4. `validate` runs `python story/_check.py` — the consistency gate (below).
5. `publish` pushes and opens a PR.
6. `review` runs `.aiorchestra/templates/review.md` — an editorial pass on the
   diff. Failures re-invoke the agent with the feedback appended.
7. **You** read the PR, edit the artifact directly if you want, and merge it.
8. **You** add `aiorchestra` + `claude` to the next phase's issue. aiorchestra
   picks it up — back to step 1.

### Why the next issue isn't auto-triggered

The agent files the next phase's issue with **`story` + `next-phase`** labels
only — *not* `aiorchestra` / `claude`. So it sits idle. That's deliberate: it
gives you a gate after every artifact (read the prose, fix continuity, change
direction) before the chain moves on. To advance, add `aiorchestra` + `claude`
to that issue.

If the agent can't run `gh` during its run, it instead writes the full
next-issue spec (title + body) into `story/STATUS.md` under a `## Next step`
heading — file it yourself with `gh issue create`.

## Clarification

If a phase's prerequisites are materially missing or self-contradictory in a way
that would force a wrong implementation, the agent emits
`NEEDS_CLARIFICATION: <question>` and makes no changes; aiorchestra posts the
question on the issue and defers it until you answer and clear the
`needs-clarification` label. The agent should *not* use this for ordinary
creative choices — it makes those itself and logs them in `story/notes.md`.

## Canon & continuity rules

(Also in `AGENTS.md` / `CLAUDE.md`.)

- The canonical story state lives in `story/`, never in issue/PR text.
- Before writing anything, read every file under `story/`.
- New canon (names, places, the rules of the world, events) is welcome, but it
  must not contradict `story/idea.md`, `story/premise.md`, `story/spine.md`,
  `story/world/*`, or earlier chapters. Log every new canonical fact and every
  rejected creative direction in `story/notes.md`.
- One issue = one phase = one artifact. Don't jump ahead.
- Chapter drafting: each chapter uses the world bible + chapter plan + enhancers
  guide + a short "story so far" recap of prior chapters for continuity. Roughly
  1 chapter in 3 (~35%) gets exactly one woven-in flourish — a long vivid
  scenery passage, a quirky but fitting object, a strange atmospheric detail, a
  revealing character tic, or a small surprising bit of background — placed
  naturally, never telegraphed.

## The validation gate (`story/_check.py`)

A light, stdlib-only consistency check, run at `validate`:

- If a downstream artifact has real content, its prerequisites must too
  (e.g. a non-empty `story/chapter_plan.md` requires non-empty
  `story/premise.md`, `story/spine.md`, and all four `story/world/*.md`).
- If `story/chapters/` contains any chapter, `story/chapter_plan.md` must be
  non-empty (and each chapter file must not be near-empty).
- If `story/final.md` exists it must be non-trivial and contain a
  `## Final Story` section.

It does *not* judge prose quality. If you've vendored `story_writer`'s
`scripts/run_qa.py` into this repo, the Assemble phase also runs it (name drift
/ character presence / cross-chapter phrase reuse) and addresses findings.

## Publishing

The `Assemble & Publish` phase builds `story/final.md`, then follows
`.claude/skills/herenow/SKILL.md` to publish it and records the resulting
`site_url` / `claim_url` in the PR body and in `story/STATUS.md`. An anonymous
publish is claimable for 24h; set `HERENOW_API_KEY` in the aiorchestra run's
environment to publish under an account. If the herenow skill isn't present in
the repo, the phase just leaves `story/final.md` and notes that publishing was
skipped.

## Prerequisites recap

- aiorchestra running somewhere with visibility into this repo, `claude-code`
  provider configured.
- `gh` available **inside the aiorchestra run** (for filing the next issue) and
  authenticated.
- For publishing: `.claude/skills/herenow/` present (the kit's init script
  vendors it) and, optionally, `HERENOW_API_KEY` in the environment.
