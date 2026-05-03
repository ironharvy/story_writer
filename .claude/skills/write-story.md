# Write Story

A multi-step interactive story writing skill that transforms a simple idea into a fully realized narrative. This mirrors the DSPy story_writer pipeline but uses Claude natively.

## Trigger

Use when the user wants to write a story, create a narrative, or says `/write-story`.

## Pipeline Overview

The story is built through 12 sequential stages — each feeds into the next. Follow every stage in order and present results to the user between stages.

1. Get the Story Idea
2. Interrogative Questions
3. Core Premise
4. Narrative Spine
5. World Bible Questions
6. World Bible
7. Chapter Plan
8. Enhancers Guide
9. Editor-in-Chief: Pre-Production Review
10. Write Chapters
11. Editor-in-Chief: Manuscript Review
12. Compile and Save

## Progress Reporting

At the start of every stage, output a single-line marker so the user can track progress in any host (Claude Code, Codex, etc.):

```
=== Stage N/12: <stage name> ===
```

In Stage 10, emit `--- Chapter K/<total>: <title> ---` before writing each chapter.

## Checkpointing

Save state often so progress survives interruptions and the user can read intermediate output at any time.

After **every stage** (and after **every chapter** in Stage 10):

1. Write `.tmp/story_state.json` containing all accumulated variables collected so far. Possible keys: `IDEA`, `QA_PAIRS`, `CORE_PREMISE`, `SPINE_TEMPLATE`, `WORLD_BIBLE_QA`, `WORLD_BIBLE`, `CHAPTER_PLAN`, `ENHANCERS_GUIDE`, `PREPRO_NOTES`, `CHAPTERS` (list of `{title, prose}`), `EDITOR_NOTES`.
2. Rebuild `.tmp/story_output.md` from current state using the format in Stage 12, omitting sections that don't yet exist. The user should always have a readable artifact on disk reflecting the latest progress.

Create `.tmp/` if it doesn't exist. Note "checkpoint saved" once per stage so the user knows.

## Instructions

### Stage 1/12: Get the Story Idea

Use `AskUserQuestion` to ask the user:
- "What is your initial story idea or premise? Describe it in a few sentences."

Store their response as `IDEA`. Checkpoint.

### Stage 2/12: Generate Interrogative Questions

Generate exactly **5 interrogative questions** that probe and challenge the user's idea to flesh it out into a full story foundation. Each question must include a **proposed answer** that you think fits the idea.

Format the questions clearly, numbered 1-5, each with:
- The question
- Your proposed answer

Present all 5 questions and proposed answers to the user. For each question, use `AskUserQuestion` to ask whether they accept the proposed answer or want to provide their own. Offer these options:
1. "Accept proposed answer"
2. "I'll provide my own answer"

If they choose to provide their own, use `AskUserQuestion` to collect it.

Store the final Q&A pairs as `QA_PAIRS`. Checkpoint.

### Stage 3/12: Generate Core Premise

Using `IDEA` and `QA_PAIRS`, synthesize a **Core Premise** — a detailed paragraph that summarizes the foundation of the story including:
- The central conflict
- The protagonist's goal and motivation
- The stakes
- The setting and tone
- The thematic undercurrent

Present the Core Premise to the user, then ask via `AskUserQuestion`:
- "Are you happy with this Core Premise?"
  - "Yes, continue"
  - "No, I want to refine it"

If they want to refine, ask what changes they'd like, regenerate incorporating their feedback, and ask again. Loop until satisfied.

Store as `CORE_PREMISE`. Checkpoint.

### Stage 4/12: Generate Narrative Spine

Using `CORE_PREMISE`, generate a **Narrative Spine Template** following the classic structure:

- **Once upon a time...** (the status quo / world setup)
- **Every day...** (the routine / normal life)
- **One day...** (the inciting incident)
- **Because of that...** (first consequence / rising action)
- **Because of that...** (escalation / complications)
- **Until finally...** (the climax / resolution)

Present the spine to the user. Store as `SPINE_TEMPLATE`. Checkpoint.

### Stage 5/12: World Bible Questions

Using `CORE_PREMISE` and `SPINE_TEMPLATE`, generate **3 follow-up questions** with proposed answers to flesh out the world-building. These should focus on:
- The rules and systems of the world (magic, technology, society)
- Key relationships and power dynamics
- Unresolved world-building details

Present and collect answers the same way as Stage 2 (accept or provide own).

Store as `WORLD_BIBLE_QA`. Checkpoint.

### Stage 6/12: Generate World Bible

Using all accumulated context (`CORE_PREMISE`, `SPINE_TEMPLATE`, `WORLD_BIBLE_QA`), generate a comprehensive **World Bible** with these four sections:

#### 6a: Rules of the World
The rules governing the story's world — magic systems, science, laws, etiquette, societal norms, and any loopholes characters might exploit.

#### 6b: Characters
Full character descriptions and biographies for every significant character:
- Full name, physical description
- Relationships to other characters
- Role, aspirations, flaws
- Main characters get detailed entries; minor characters get brief ones

#### 6c: Locations
All significant places in the story:
- Physical description and climate
- Who lives/works there
- Geographic relationships to other locations
- Atmosphere and significance to the plot

#### 6d: Plot Timeline
A chronological sequence of major events, from backstory through the story's conclusion.

Present the complete World Bible to the user. Store as `WORLD_BIBLE`. Checkpoint.

### Stage 7/12: Generate Chapter Plan

Using `CORE_PREMISE` and `WORLD_BIBLE`, generate a chapter plan across **3 acts**:

- **Act 1 (Setup):** Introduce characters, world, and inciting incident
- **Act 2 (Confrontation):** Rising action, complications, midpoint shift
- **Act 3 (Resolution):** Climax, falling action, resolution

For each act, generate 3-5 chapter descriptions (so roughly 9-15 chapters total). Each chapter description should be a concise sentence describing the key event/purpose of that chapter.

Present the full chapter plan to the user. Store as `CHAPTER_PLAN`. Checkpoint.

### Stage 8/12: Generate Enhancers Guide

Using `WORLD_BIBLE` and `CHAPTER_PLAN`, evaluate which **story enhancers** should be applied to specific chapters:

- **Tension Module** — where to build and release tension
- **Mystery Module** — where to plant questions and reveal answers
- **Theme Alignment** — where themes should surface
- **Setup/Payoff Tracker** — what setups need payoffs and where
- **Emotional Curve** — the emotional trajectory across chapters
- **Twist Generator** — where surprises or reversals should land
- **Easter Egg Injector** — subtle callbacks or hidden connections

Present the enhancers guide. Store as `ENHANCERS_GUIDE`. Checkpoint.

### Stage 9/12: Editor-in-Chief — Pre-Production Review

Before any prose is written, do a critical pass over everything assembled so far: `CORE_PREMISE`, `SPINE_TEMPLATE`, `WORLD_BIBLE`, `CHAPTER_PLAN`, and `ENHANCERS_GUIDE`. Adopt the voice of a hard-nosed editor-in-chief — your job is to find problems now, while they're cheap to fix.

Look for:
- **Plot holes & logic gaps** — does the chapter plan actually flow from the premise? Are there unmotivated jumps between chapters?
- **Character motivation** — does the protagonist have a clear arc? Are antagonists' goals coherent and consistent with their established traits?
- **World-rule consistency** — does the chapter plan respect the rules established in the World Bible? Any chapter rely on something the world doesn't allow?
- **Setup/payoff balance** — every setup in the Enhancers Guide should have a planned payoff in `CHAPTER_PLAN`, and every major payoff should have a setup. Flag orphans on either side.
- **Pacing** — is Act 2 doing real work or sagging? Is the climax positioned correctly? Any act feel underweight?
- **Theme alignment** — is the thematic undercurrent surfaced across the chapter plan, not just declared once?
- **Stakes escalation** — do the stakes meaningfully rise across acts, or stay flat?

Produce a numbered list of notes. If everything checks out, say "no notes" — don't manufacture problems. Present the notes to the user.

Use `AskUserQuestion`:
- "How should I act on these notes?"
  - "Apply all"
  - "Apply selected (tell me which numbers)"
  - "Ignore — keep current plan"

If applying, regenerate the affected artifacts (any of `CORE_PREMISE`, `WORLD_BIBLE`, `CHAPTER_PLAN`, `ENHANCERS_GUIDE`) incorporating the accepted notes, present the updated versions, and overwrite the same variables.

Store the notes (and which were applied) as `PREPRO_NOTES`. Checkpoint.

### Stage 10/12: Write the Story

Now write each chapter one at a time. For each chapter:

1. **Progress marker:** emit `--- Chapter K/<total>: <title> ---` before writing.

2. **Random Detail Injection (35% chance per chapter):** Roll a mental dice. Roughly 1 in 3 chapters should receive a creative flourish — one of these types:
   - A vivid, unusually long description of scenery or environment
   - A quirky or unexpected object placed naturally in the scene
   - A strange but fitting atmospheric detail (sounds, smells, textures)
   - An unusual yet revealing character habit, tic, or physical detail
   - A brief, surprising background element enriching the world

3. **Write the chapter** using:
   - `WORLD_BIBLE` for consistency
   - `CHAPTER_PLAN` for structure
   - `ENHANCERS_GUIDE` for what narrative techniques to apply
   - Summary of previous chapters for continuity
   - The random detail (if triggered) woven naturally into the prose

4. Each chapter should include:
   - A creative chapter title
   - Rich, immersive prose with dialogue and description
   - Consistent characterization and world details
   - Natural pacing appropriate to its position in the story

5. **Checkpoint immediately** after the chapter — append `{title, prose}` to `CHAPTERS`, then write `.tmp/story_state.json` and rebuild `.tmp/story_output.md`. This way an interruption at chapter 8 of 12 doesn't lose chapters 1-7.

Do NOT ask for confirmation between chapters — write them all in sequence.

### Stage 11/12: Editor-in-Chief — Manuscript Review

Now critique the actual prose. Read across all chapters and build a numbered notes list covering:

- **Continuity breaks** — names, ages, locations, world rules contradicted between chapters
- **Character voice drift** — does each character sound consistent across chapters?
- **Unfired setups** — anything from `ENHANCERS_GUIDE` or earlier chapters that lacks payoff in the prose as written
- **Pacing dead spots** — chapters that don't move the plot or reveal character
- **Repetition** — repeated phrases, near-duplicate sentences, overused words. (Mirrors the spirit of `qa.py`'s sentence-similarity check — flag any near-duplicates you notice.)
- **Show vs. tell** — exposition dumps that should have been scenes
- **Climax & resolution** — does the ending land? Are loose threads tied?

Present numbered notes to the user. Say "no notes" if there are none.

Use `AskUserQuestion`:
- "How should I act on these notes?"
  - "Apply all"
  - "Apply selected (tell me which numbers)"
  - "Ignore — keep manuscript as-is"

For accepted notes, revise the affected chapters in place, replacing entries in `CHAPTERS`. Briefly summarize what was changed per chapter. Checkpoint after each revision.

Store the notes as `EDITOR_NOTES`.

### Stage 12/12: Compile and Save Output

Rebuild `.tmp/story_output.md` one last time as the canonical artifact:

```markdown
# Story Output

## Core Premise
{CORE_PREMISE}

## Spine Template
{SPINE_TEMPLATE}

## World Bible
{WORLD_BIBLE}

## Chapter Plan
{CHAPTER_PLAN}

## Enhancers Guide
{ENHANCERS_GUIDE}

## Pre-Production Editor Notes
{PREPRO_NOTES}

## Final Story
{All chapters — title + prose}

## Manuscript Editor Notes
{EDITOR_NOTES}
```

Confirm the file path to the user.

## Important Guidelines

- **Be immersive and literary** — this is creative writing, not a summary. Chapters should be full prose with dialogue, description, and internal thought.
- **Maintain consistency** — character names, world rules, and established facts must stay consistent throughout.
- **Show, don't tell** — use scenes and dialogue to convey information rather than exposition dumps.
- **Each chapter should be substantial** — aim for rich, detailed writing (not just a paragraph per chapter).
- **Respect user choices** — the user's answers to questions override your proposed answers. Build the story around their vision, not yours.
- **Keep the user informed** — at each stage, clearly present what was generated before moving to the next stage.
- **Save often** — checkpoint after every stage and every chapter. The user should never lose more than one chapter's worth of work to an interruption.
- **Editor passes are honest, not performative** — if there are no real problems, say "no notes" and move on. Don't invent issues to look thorough.
