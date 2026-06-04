You are developing a story in this repository, one phase at a time, driven by aiorchestra. Implement the following GitHub issue — and only this issue.

Issue #{number}: {title}

{body}
{osint_context}{comments_section}

## First, always

1. Read `WORKFLOW.md` (repo root), `AGENTS.md`, and `CLAUDE.md`.
2. Read **every** file under `story/` — the canonical state of the story lives there, not in issues or PRs.
3. Identify which phase this is from the issue-title prefix (table below). Do that phase and nothing else: one issue = one phase = exactly one new/updated story artifact (plus `story/STATUS.md`, plus — for non-terminal phases — filing the next issue).

Do NOT run tests — aiorchestra runs the validation gate (`python story/_check.py`) separately.

## Phase routing — by issue-title prefix

- `Idea:` → Refine `story/idea.md`. It must contain: the seed idea (already present under `## Seed (from README)`), then exactly **5 interrogative questions** that probe and sharpen the idea, each with a **proposed answer** and an **accepted answer** (use the proposed answer unless the issue body overrides it; if the issue body supplies answers, use those), and enough foundation for the Premise phase. Also append a `## Style` section to `AGENTS.md` *and* `CLAUDE.md`: prose register, POV, tense, tone, length target (short story vs. novel), and any stylistic constraints implied by the idea — every later phase relies on it.
- `Premise:` → Write `story/premise.md`: one tight **Core Premise** paragraph covering the central conflict, the protagonist's goal and motivation, the stakes, the setting and tone, and the thematic undercurrent. Built from `story/idea.md`.
- `Spine:` → Write `story/spine.md`: the six-beat template, one to three sentences per beat — `Once upon a time…` (status quo / world), `Every day…` (the routine), `One day…` (inciting incident), `Because of that…` (first consequence), `Because of that…` (escalation / complications), `Until finally…` (climax / resolution). Abstract; names not required yet. Built from `story/premise.md`.
- `World Bible — Rules:` → Write `story/world/rules.md`: the rules and systems governing the world — magic, technology, science, law, etiquette, societal norms, and any loopholes characters might exploit. Built from `story/premise.md` + `story/spine.md`.
- `World Bible — Characters:` → Write `story/world/characters.md`: every significant character — full name, physical description, relationships to other characters, role, aspirations, flaws. Major characters get detailed entries; minor ones get a line or two. Built from the premise, the spine, and `story/world/rules.md`.
- `World Bible — Locations:` → Write `story/world/locations.md`: every significant place — physical description and climate, who lives/works there, geographic relationship to other places, atmosphere, significance to the plot. Built from the above plus `story/world/characters.md`.
- `World Bible — Timeline:` → Write `story/world/timeline.md`: a chronological sequence of major events, from backstory through the story's conclusion. Built from all of `story/world/`.
- `Chapter Plan:` → Write `story/chapter_plan.md`: a plan across **three acts** — Act 1 (Setup: characters, world, inciting incident), Act 2 (Confrontation: rising action, complications, midpoint shift), Act 3 (Resolution: climax, falling action, resolution). 3-5 chapters per act (~9-15 total). One concise sentence per chapter describing its key event/purpose, numbered consecutively. Built from `story/premise.md` + `story/world/`.
- `Enhancers:` → Write `story/enhancers.md`: for each chapter in `story/chapter_plan.md`, note which narrative enhancers apply and how — Tension (build/release points), Mystery (where to plant questions, where to reveal answers), Theme alignment (where themes surface), Setup/Payoff tracker (what setups need payoffs, and in which chapter), Emotional curve (the emotional trajectory across chapters), Twist generator (where reversals land), Easter-egg injector (subtle callbacks / hidden connections). Built from `story/chapter_plan.md` + `story/world/`.
- `Chapters:` → Draft **every** chapter listed in `story/chapter_plan.md`, in order, each to its own file `story/chapters/NN-slug.md` (zero-padded number, e.g. `01-the-fog-comes-in.md`). For each chapter use the world bible, the chapter plan, the enhancers guide, and a short "story so far" recap of the chapters you have already written this run for continuity. Roughly 1 chapter in 3 (~35%) gets exactly one woven-in flourish — a long vivid scenery passage, a quirky but fitting object, a strange atmospheric detail, a revealing character tic, or a small surprising bit of background — placed naturally, never telegraphed. Each chapter: a creative title, full immersive prose with dialogue and description, consistent characterisation and world detail, pacing suited to its position in the arc. Substantial chapters, not paragraph summaries. *If `story/idea.md` clearly calls for a long novel rather than a short story, do NOT draft all chapters here — instead see "Advancing the chain" and file one `Chapter NN:` issue per chapter.*
- `Chapter NN:` → Draft just chapter `NN` from `story/chapter_plan.md` to `story/chapters/NN-slug.md`, same rules as above, using the already-written chapters in `story/chapters/` for continuity.
- `Assemble & Publish:` → Build `story/final.md` with these sections, in order: a `# <story title>` heading, `## Core Premise` (from `story/premise.md`), `## Spine` (from `story/spine.md`), `## World Bible` (Rules / Characters / Locations / Timeline subsections from `story/world/`), `## Chapter Plan` (from `story/chapter_plan.md`), `## Enhancers Guide` (from `story/enhancers.md`), `## Final Story` (every chapter, in order, with its title and full prose). If `scripts/run_qa.py` exists in this repo, run it on `story/final.md` and fix any clear issues it reports (name drift, missing characters, cross-chapter phrase reuse) before finishing. Then publish: follow `.claude/skills/herenow/SKILL.md` to publish `story/final.md`, and record the returned `site_url` and `claim_url` in `story/STATUS.md` and in your final summary (which becomes the PR body). If `.claude/skills/herenow/` is not present, skip publishing and note that in `story/STATUS.md`. This is the terminal phase — do NOT file another issue.
- `Revise — <path>:` → Edit only the file named in the title (e.g. `Revise — story/chapters/03-the-mainland-letter.md: pacing drags mid-scene`), following the issue body's notes, keeping everything consistent with the rest of `story/`. Do NOT file another issue; this is a touch-up, not a chain step.

If the title doesn't match any prefix above, treat the issue body as the spec, change as little as possible, and do not file a follow-up issue.

## Standing rules

- Never contradict established canon: `story/idea.md`, `story/premise.md`, `story/spine.md`, `story/world/*`, and any already-written chapters. New canon is fine and encouraged, but it must be consistent — and every new canonical fact, plus every creative direction you considered and rejected, goes in `story/notes.md` (append, with a short issue/date note).
- Update `story/STATUS.md`: mark this phase done (note "PR pending"), and keep the table accurate.
- Honour the `## Style` section in `AGENTS.md`.
- Keep prose for the reader: no meta commentary about phases, issues, or PRs inside the story files — `story/STATUS.md` and `story/notes.md` are the place for process notes.

## Advancing the chain

When you finish a phase that is **not** `Assemble & Publish:` and **not** `Revise — …:`, file the next phase's issue yourself:

1. Determine the next issue's title from the chain in `WORKFLOW.md` (e.g. after `Premise:` comes `Spine: <same story title>`; after the `Chapters:` phase comes `Assemble & Publish: <story title>`). If `story/idea.md` calls for a long novel, the chain after `Enhancers:` is `Chapter 01: …`, `Chapter 02: …`, … (titles from `story/chapter_plan.md`), then `Assemble & Publish:`.
2. Build the body from the matching file in `.github/ISSUE_TEMPLATE/` — strip its YAML front matter, then fill in the story title and a short "where we are" pointer listing the relevant `story/` files the next phase should read.
3. Create it with the `gh` CLI, labelled `story` and `next-phase` **only** — never add `aiorchestra` or `claude` (a human adds those after reviewing this PR; that is the gate between phases). For example: `gh issue create --title "Spine: <story title>" --body-file <tmpfile> --label story --label next-phase`. Record the new issue's number/URL in your final summary.
4. If `gh` is unavailable or fails, do not block: write the full next-issue spec (title + body) into `story/STATUS.md` under a `## Next step` heading so a human can file it.

## Clarification protocol

If this issue's prerequisites are materially missing or self-contradictory in a way that would force a wrong implementation, do NOT guess. Output a single line:

    NEEDS_CLARIFICATION: <your question>

and make NO file changes — aiorchestra will post the question on the issue and pause until a human answers. Use this only for genuine blockers; resolve ordinary creative choices yourself and log them in `story/notes.md`.
