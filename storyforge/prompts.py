"""Prompt builders for each pipeline stage.

Every prompt bakes in a defense against a specific eval-spec failure mode:
named protagonist from the start (T2.2), single locked POV (T2.1), distinct
spellable names (T1.3/T2.3), full-cast coverage (T1.2), word budgets (T1.1),
dramatize-don't-summarize + fresh diction (T1.4/T1.5/rubric).
"""
from __future__ import annotations

from .models import ChapterPlan, Premise, StorySpec, WorldBible

GEN_SYSTEM = (
    "You are a master novelist and story architect. You write vivid, economical "
    "prose and plan stories with rigorous structure. You follow instructions about "
    "point of view, character names, and length exactly, and you never pad."
)


def infer_spec_prompt(idea: str, provided: dict) -> str:
    return (
        f"A writer gave this story idea:\n\"\"\"\n{idea}\n\"\"\"\n\n"
        "Infer sensible defaults for any unspecified production choices. Already "
        f"chosen by the writer (keep these): {provided or 'none'}.\n\n"
        'Return JSON: {"genre": "...", "tone": "...", '
        '"pov": "first_person" | "third_person_limited" | "third_person_omniscient", '
        '"ending": "<one phrase describing the kind of ending that best fits>", '
        '"audience": "..."}'
    )


def premise_prompt(idea: str, spec: StorySpec) -> str:
    return (
        f"Story idea:\n\"\"\"\n{idea}\n\"\"\"\n\n"
        f"Genre: {spec.genre}. Tone: {spec.tone}. POV: {spec.pov_label}. "
        f"Desired ending: {spec.ending}.\n\n"
        "Forge the foundation. The protagonist MUST be given a concrete proper name "
        "now (it will be used from the first sentence of chapter one).\n\n"
        'Return JSON: {"title": "<evocative title>", '
        '"logline": "<<=25 words: names the protagonist, their goal, the conflict, the stakes>", '
        '"paragraph": "<one paragraph, 4-6 sentences: ordinary world + inciting incident, '
        'the rising disasters and midpoint turn, and how it resolves>", '
        '"theme": "<one sentence: the human question the story explores>"}'
    )


def bible_prompt(idea: str, spec: StorySpec, premise: Premise) -> str:
    return (
        f"Idea: {idea}\nTitle: {premise.title}\nLogline: {premise.logline}\n"
        f"Paragraph: {premise.paragraph}\nTheme: {premise.theme}\n"
        f"Genre/tone: {spec.genre}/{spec.tone}. POV: {spec.pov_label}.\n\n"
        "Build the story bible.\n"
        'Return JSON: {"characters": [{"name": "...", "role": '
        '"protagonist|antagonist|ally|mentor|foil|...", "description": "<1 line>", '
        '"want": "<external goal>", "need": "<internal lesson>", "arc": "<how they change>"}], '
        '"locations": [{"name": "...", "description": "<1 line>"}], '
        '"rules": ["<key world constraint the plot must obey>"]}\n\n'
        "Constraints:\n"
        "- 3 to 6 characters. Exactly ONE has role 'protagonist'.\n"
        "- Give every character a DISTINCT, easy-to-spell name: no two names sharing "
        "the same first syllable, no near-duplicates, no hyphenated names, no titles "
        "alone. These spellings are canonical and must never drift.\n"
        "- 2 to 5 locations. 2 to 5 concrete world rules.\n"
        "- The protagonist's want and need must be in tension (that tension is the arc)."
    )


def spine_prompt(spec: StorySpec, premise: Premise, bible: WorldBible) -> str:
    cast = "; ".join(f"{c.name} ({c.role})" for c in bible.characters)
    protagonist = next((c.name for c in bible.characters if c.role == "protagonist"),
                       spec.pov_character or (bible.names()[0] if bible.names() else "the protagonist"))
    return (
        f"Title: {premise.title}\nLogline: {premise.logline}\nTheme: {premise.theme}\n"
        f"Paragraph: {premise.paragraph}\nCast: {cast}\n"
        f"POV: {spec.pov_label}, narrated by {protagonist} in EVERY chapter.\n\n"
        f"Plan exactly {spec.num_chapters} chapters as a three-act arc with a clear "
        "climax in the final third and an earned ending that pays off the theme and the "
        "protagonist's need.\n\n"
        'Return JSON: {"chapters": [{"number": <int>, "title": "...", '
        f'"pov_character": "{protagonist}", '
        '"summary": "<2-3 sentences of what happens>", '
        '"beats": ["<3-5 scene/sequel beats: a goal pursued, conflict, and the turn>"], '
        '"goal": "<the POV character\'s goal this chapter>", '
        '"conflict": "<what opposes it>", '
        '"outcome": "<how it ends and shifts want/need>", '
        '"characters_present": ["<names>"], '
        f'"word_target": {spec.words_per_chapter}}}]}}\n\n'
        "Constraints:\n"
        f"- Exactly {spec.num_chapters} chapters, numbered 1..{spec.num_chapters}.\n"
        "- EVERY named character in the cast must appear in characters_present of at "
        "least one chapter.\n"
        "- Events must follow cause->effect; the protagonist's CHOICES (not luck) drive "
        "each turn. The penultimate chapter is the lowest point / hardest choice; the "
        "final chapter is climax + resolution (no new characters, no deus ex machina)."
    )


def _cast_brief(bible: WorldBible) -> str:
    return "\n".join(f"- {c.name}: {c.description} (wants {c.want}; needs {c.need})"
                     for c in bible.characters)


def draft_chapter_prompt(spec: StorySpec, premise: Premise, bible: WorldBible,
                         plan: ChapterPlan, synopsis: str, prev_tail: str,
                         avoid: list[str]) -> str:
    rules = "; ".join(bible.rules) if bible.rules else "(none)"
    avoid_block = ""
    if avoid:
        avoid_block = ("\nDo NOT reuse these phrasings from earlier chapters — invent "
                       "fresh imagery:\n" + "\n".join(f'  - "{a}"' for a in avoid))
    context = ""
    if synopsis:
        context += f"\nStory so far:\n{synopsis}"
    if prev_tail:
        context += f"\nThe previous chapter ended:\n\"…{prev_tail}\""
    return (
        f"Title: {premise.title}\nLogline: {premise.logline}\nTheme: {premise.theme}\n"
        f"World rules: {rules}\nCast:\n{_cast_brief(bible)}\n"
        f"{context}\n{avoid_block}\n\n"
        f"=== Write Chapter {plan.number}: {plan.title} ===\n"
        f"Plan — summary: {plan.summary}\n"
        f"goal: {plan.goal} | conflict: {plan.conflict} | outcome: {plan.outcome}\n"
        f"beats: {' / '.join(plan.beats)}\n"
        f"characters present: {', '.join(plan.characters_present) or 'as needed'}\n\n"
        "Write the chapter prose now. Rules:\n"
        f"- Output ONLY the prose. No heading, no chapter number, no notes, no markdown.\n"
        f"- Point of view: {spec.pov_label}; the narrating consciousness is "
        f"{plan.pov_character}. Never shift POV within the chapter.\n"
        f"- Refer to {plan.pov_character} by name. NEVER use placeholders like 'the "
        "protagonist', 'the boy', 'the girl', 'the woman', 'the man'.\n"
        f"- Length: about {plan.word_target} words (no fewer than 800).\n"
        "- DRAMATIZE in scene: concrete action, dialogue, sensory detail, interiority. "
        "Minimize summary narration.\n"
        "- Keep every fact consistent with the story so far and the cast/world rules.\n"
        "- Vary sentence openings and word choice; avoid verbal tics and repeated "
        "imagery.\n"
        "- Do NOT restate the premise or re-explain the world's central rules; assume "
        "the reader remembers them, and refer to recurring elements with fresh wording "
        "every chapter.\n"
        "- Advance goal->conflict->outcome through the character's choices, and end on a "
        "turn or hook."
    )


def summary_prompt(plan: ChapterPlan, prose: str) -> str:
    return (
        f"Chapter {plan.number}: {plan.title}\n\n{prose}\n\n"
        "Summarize this chapter for a continuity bible.\n"
        'Return JSON: {"summary": "<2-3 sentences: what changed in plot AND character>", '
        '"facts": ["<new canonical facts: deaths, reveals, relationships, locations>"]}'
    )


def critique_prompt(spec: StorySpec, plan: ChapterPlan, synopsis: str, prose: str) -> str:
    return (
        f"You are a strict continuity + line editor. POV must be {spec.pov_label} "
        f"narrated by {plan.pov_character}.\nStory so far:\n{synopsis or '(this is chapter 1)'}\n\n"
        f"Chapter {plan.number} draft:\n\"\"\"\n{prose}\n\"\"\"\n\n"
        "Report ONLY serious problems: POV slips; the protagonist referred to by a "
        "placeholder instead of their name; facts that contradict the story so far; "
        "length under ~800 words; or a vivid phrase/simile repeated from earlier "
        "chapters.\n"
        'Return JSON: {"ok": true|false, "issues": ["..."], '
        '"fixes": ["<concrete fix instruction>"]}. If the draft is solid, ok=true and '
        "empty lists."
    )


def polish_chapter_prompt(spec: StorySpec, premise: Premise, bible: WorldBible,
                          plan: ChapterPlan, prose: str, synopsis: str,
                          notes: list[str], is_climax: bool) -> str:
    rules = "; ".join(bible.rules) if bible.rules else "(none)"
    note_block = "\n".join(f"  - {n}" for n in notes)
    climax = ("\n  - This is the CLIMAX: make the confrontation vivid, high-stakes and "
              "decisive. Do not rush it; give the turning point room to land."
              if is_climax else "")
    return (
        f"You are doing a developmental + line edit of Chapter {plan.number} of a "
        f"finished manuscript titled “{premise.title}”.\n"
        f"Theme: {premise.theme}\nWorld rules (keep them CONSISTENT): {rules}\n"
        f"Cast: {_cast_brief(bible)}\n"
        f"Story so far (for cohesion): {synopsis or '(opening chapter)'}\n\n"
        "Editorial notes to address in this chapter:\n" + note_block + climax + "\n\n"
        f"Chapter {plan.number}: {plan.title}\n\"\"\"\n{prose}\n\"\"\"\n\n"
        "Revise the chapter to fix the notes while PRESERVING: the point of view "
        f"({spec.pov_label}, narrated by {plan.pov_character}), every plot fact, all "
        "character names, and the chapter's role in the arc. Cut meandering or "
        "repetitive passages; vary imagery and sentence rhythm; remove clichés and any "
        "phrase you have used elsewhere. Keep length within ~15% of the original.\n"
        "Output ONLY the revised chapter prose — no heading, no notes, no markdown."
    )


def derepeat_prompt(number: int, prose: str, phrases: list[str]) -> str:
    return (
        f"This is Chapter {number}. The phrasings below were ALSO used in other "
        "chapters of the same book — that repetition reads as a verbal tic. Rewrite "
        "the chapter so each of these ideas is expressed with COMPLETELY fresh wording "
        "(different words and sentence structure), while keeping the same events, "
        "point of view, characters, facts, and meaning. Do not shorten the chapter.\n\n"
        "Phrasings to rephrase (do not reuse these word sequences):\n"
        + "\n".join(f'  - "{p}"' for p in phrases)
        + f"\n\nChapter {number} draft:\n\"\"\"\n{prose}\n\"\"\"\n\n"
        "Output ONLY the revised chapter prose — no heading, no notes, no markdown."
    )


def revise_prompt(plan: ChapterPlan, prose: str, fixes: list[str]) -> str:
    return (
        f"Revise Chapter {plan.number} to fix these problems while keeping what works:\n"
        + "\n".join(f"- {f}" for f in fixes)
        + f"\n\nCurrent draft:\n\"\"\"\n{prose}\n\"\"\"\n\n"
        "Output ONLY the revised chapter prose — no heading, no notes, no markdown."
    )
