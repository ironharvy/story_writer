# Write Story (Modified for AI Agent)

A multi-step interactive story writing skill that transforms a simple idea into a fully realized narrative. This mirrors the DSPy story_writer pipeline but uses the AI agent's available tools.

## Trigger

Use when the user wants to write a story, create a narrative, or says `/write-story`.

## Pipeline Overview

The story is built through these sequential stages — each stage feeds into the next. You MUST follow every stage in order and present results to the user between stages.

## Available Tools

- `question`: For interactive user questions with multiple choice options
- `write`: To save the final output to `.tmp/story_output.md`
- `bash`: To create directories if needed
- All other standard tools for reading/editing files

## Instructions

### Stage 1: Get the Story Idea

Use `question` tool to ask the user:
- "What is your initial story idea or premise? Describe it in a few sentences."

Store their response as `IDEA`.

### Stage 2: Generate Interrogative Questions

Generate exactly **5 interrogative questions** that probe and challenge the user's idea to flesh it out into a full story foundation. Each question must include a **proposed answer** that you think fits the idea.

Format the questions clearly, numbered 1-5, each with:
- The question
- Your proposed answer

Present all 5 questions and proposed answers to the user. For each question, use `question` tool to ask whether they accept the proposed answer or want to provide their own. Offer these options:
1. "Accept proposed answer"
2. "I'll provide my own answer"

If they choose to provide their own, use `question` tool with `custom: true` to collect it.

Store the final Q&A pairs as `QA_PAIRS`.

### Stage 3: Generate Core Premise

Using `IDEA` and `QA_PAIRS`, synthesize a **Core Premise** — a detailed paragraph that summarizes the foundation of the story including:
- The central conflict
- The protagonist's goal and motivation
- The stakes
- The setting and tone
- The thematic undercurrent

Present the Core Premise to the user, then ask via `question` tool:
- "Are you happy with this Core Premise?"
  - "Yes, continue" 
  - "No, I want to refine it"

If they want to refine, ask what changes they'd like (use `question` with `custom: true`), regenerate incorporating their feedback, and ask again. Loop until satisfied.

Store as `CORE_PREMISE`.

### Stage 4: Generate Narrative Spine

Using `CORE_PREMISE`, generate a **Narrative Spine Template** following the classic structure:

- **Once upon a time...** (the status quo / world setup)
- **Every day...** (the routine / normal life)
- **One day...** (the inciting incident)
- **Because of that...** (first consequence / rising action)
- **Because of that...** (escalation / complications)
- **Until finally...** (the climax / resolution)

Present the spine to the user. Store as `SPINE_TEMPLATE`.

### Stage 5: World Bible Questions

Using `CORE_PREMISE` and `SPINE_TEMPLATE`, generate **3 follow-up questions** with proposed answers to flesh out the world-building. These should focus on:
- The rules and systems of the world (magic, technology, society)
- Key relationships and power dynamics
- Unresolved world-building details

Present and collect answers the same way as Stage 2 (accept or provide own).

Store as `WORLD_BIBLE_QA`.

### Stage 6: Generate World Bible

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

Present the complete World Bible to the user. Store as `WORLD_BIBLE`.

### Stage 7: Generate Chapter Plan

Using `CORE_PREMISE` and `WORLD_BIBLE`, generate a chapter plan across **3 acts**:

- **Act 1 (Setup):** Introduce characters, world, and inciting incident
- **Act 2 (Confrontation):** Rising action, complications, midpoint shift
- **Act 3 (Resolution):** Climax, falling action, resolution

For each act, generate 3-5 chapter descriptions (so roughly 9-15 chapters total). Each chapter description should be a concise sentence describing the key event/purpose of that chapter.

Present the full chapter plan to the user. Store as `CHAPTER_PLAN`.

### Stage 8: Generate Enhancers Guide

Using `WORLD_BIBLE` and `CHAPTER_PLAN`, evaluate which **story enhancers** should be applied to specific chapters:

- **Tension Module** — where to build and release tension
- **Mystery Module** — where to plant questions and reveal answers
- **Theme Alignment** — where themes should surface
- **Setup/Payoff Tracker** — what setups need payoffs and where
- **Emotional Curve** — the emotional trajectory across chapters
- **Twist Generator** — where surprises or reversals should land
- **Easter Egg Injector** — subtle callbacks or hidden connections

Present the enhancers guide. Store as `ENHANCERS_GUIDE`.

### Stage 9: Write the Story

Now write each chapter one at a time. For each chapter:

1. **Random Detail Injection (35% chance per chapter):** Roll a mental dice. Roughly 1 in 3 chapters should receive a creative flourish — one of these types:
   - A vivid, unusually long description of scenery or environment
   - A quirky or unexpected object placed naturally in the scene
   - A strange but fitting atmospheric detail (sounds, smells, textures)
   - An unusual yet revealing character habit, tic, or physical detail
   - A brief, surprising background element enriching the world

2. **Write the chapter** using:
   - `WORLD_BIBLE` for consistency
   - `CHAPTER_PLAN` for structure
   - `ENHANCERS_GUIDE` for what narrative techniques to apply
   - Summary of previous chapters for continuity
   - The random detail (if triggered) woven naturally into the prose

3. Each chapter should include:
   - A creative chapter title
   - Rich, immersive prose with dialogue and description
   - Consistent characterization and world details
   - Natural pacing appropriate to its position in the story

After writing each chapter, briefly note which chapter you just completed, then continue to the next. Do NOT ask for confirmation between chapters — write them all in sequence.

### Stage 10: Compile and Save Output

After all chapters are written, compile everything into a single markdown file at `.tmp/story_output.md`:

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

## Final Story
{All chapters with titles and prose}
```

Use `bash` to create the `.tmp` directory if it doesn't exist. Use `write` to save the file. Tell the user the file path when done.

## Important Guidelines

- **Be immersive and literary** — this is creative writing, not a summary. Chapters should be full prose with dialogue, description, and internal thought.
- **Maintain consistency** — character names, world rules, and established facts must stay consistent throughout.
- **Show, don't tell** — use scenes and dialogue to convey information rather than exposition dumps.
- **Each chapter should be substantial** — aim for rich, detailed writing (not just a paragraph per chapter).
- **Respect user choices** — the user's answers to questions override your proposed answers. Build the story around their vision, not yours.
- **Keep the user informed** — at each stage, clearly present what was generated before moving to the next stage.
- **Use markdown formatting** for headings, emphasis, and structure in the final output.
