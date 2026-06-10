"""Variant D — self-reviewing drafter with per-chapter LLM gates.

Each chapter is drafted, then put through fast per-chapter gates (word-count
floor, bible consistency, reading-comprehension self-review). Blocking findings
plus a length shortfall become a revision brief that is fed straight back into
the drafter, up to ``max_revisions`` times. After all chapters land, two
once-per-manuscript gates run with the model's reasoning enabled — cross-chapter
continuity and premise fidelity — and their findings are written into the
artifact as a Review Report.

Unlike the baseline, nothing here blocks on human input: it is built to run
headless to completion. The rolling ``story_so_far`` summary from
:mod:`story` is the carry between chapters.
"""
from __future__ import annotations

import logging

import dspy

from core.artifact import update_artifact
from core.foundation import act_hint_for_chapter
from core.types import (
    DraftedChapter,
    DraftingInput,
    DraftingOutput,
    WorldBible,
)
from evals import (
    bible_consistency,
    bible_internal,
    comprehension,
    continuity,
    phrase_freshness,
    premise_fidelity,
)
from evals.types import EvalFinding, GateReport
from generators import register
from story import run_generate_story_so_far

logger = logging.getLogger(__name__)


class UpdateContinuityLedger(dspy.Signature):
    """Maintain a terse, authoritative ledger of continuity-critical facts after
    a chapter is written, so later chapters never contradict earlier ones.

    Carry forward the prior ledger and fold in what THIS chapter establishes or
    changes. Track, as short bullet lines: each named character's status (alive
    / dead / incapacitated / missing), their chronological age and any alias or
    title, who is related to whom, and load-bearing possessions or locations.
    When a character dies, mark them dead and keep them dead. Distinguish two
    different characters who share a name. Keep it under ~25 lines; drop
    one-off details that no longer matter.
    """

    prior_ledger: str = dspy.InputField(desc="The continuity ledger so far; empty at the start")
    chapter_title: str = dspy.InputField()
    chapter_prose: str = dspy.InputField()
    ledger: str = dspy.OutputField(desc="Updated authoritative continuity ledger")


class DraftChapterReviewed(dspy.Signature):
    """Write the chapter prose as a concrete, dramatized scene.

The world bible (locations, characters, timeline, rules) is reference material,
not source material — describe scenes in fresh language and never quote
evocative phrases from the bible verbatim. Honour the bible's facts: when a
named location or character appears, portray it as the bible defines it.

Drive the chapter on a real scene with friction (a request denied, a plan
reversed, a price paid on the page). When a character first appears, name them.
Whenever the world's defining mechanic exacts a cost (for a hard-magic premise:
blood spilled, years burned off a body), SHOW that cost physically — on flesh,
on an object, in the room — rather than asserting it. Ground every scene in
specific sensory detail of its location. End on an action or image, not a
thematic aphorism. Vary sentence rhythm; avoid stacking triadic 'X and Y'
constructions or 'not X but Y' parallelism.

Write in a tone appropriate to {act_hint}; do not foreshadow later acts. The
story_spine covers beats only up to the current act — treat anything beyond it
as not yet decided. Aim for approximately {target_words} words of prose; reach
it by deepening scene, interiority, and consequence, never by padding or
repetition.

When revision_feedback is non-empty, it lists concrete problems a reviewer
found in your previous draft of THIS chapter. Rewrite the chapter to fix every
item while keeping what already worked.
"""

    story_idea: str = dspy.InputField()
    story_title: str = dspy.InputField()
    story_spine: str = dspy.InputField()
    act_hint: str = dspy.InputField(desc="Which act of the three-act structure")
    rules_of_the_world: list[str] = dspy.InputField()
    characters: list[str] = dspy.InputField()
    locations: list[str] = dspy.InputField()
    timeline: list[str] = dspy.InputField()
    chapter: str = dspy.InputField(desc="This chapter's title and beats")
    continuity_ledger: str = dspy.InputField(
        desc="Hard facts that must stay true: which characters are alive vs dead, "
        "their chronological ages and aliases, key possessions. Do NOT contradict "
        "this — do not revive the dead, rename a character, or flip a stated age. "
        "Empty for the first chapter.",
        default="",
    )
    canon_notes: str = dspy.InputField(
        desc="Authoritative rulings that resolve bible contradictions; obey them "
        "exactly (e.g. which surname siblings share). Empty if none.",
        default="",
    )
    story_so_far: str = dspy.InputField(
        desc="Continuity context only — what happened so far. Reference material, "
        "not source: do not mirror its terse recap voice. Write fresh scene prose."
    )
    target_words: int = dspy.InputField()
    revision_feedback: str = dspy.InputField(
        desc="Reviewer problems to fix from the previous draft; empty on first pass",
        default="",
    )
    prose: str = dspy.OutputField(desc="Chapter prose")


def _word_count(text: str) -> int:
    return len(text.split())


def _premise_evidence(chapters: list[DraftedChapter], words_per_chapter: int = 140) -> str:
    """Build a bounded prose-excerpt digest for premise-fidelity judging.

    Openings carry the dramatized scene-setting and stakes, so the first
    ~``words_per_chapter`` words of each chapter are a far better signal of
    whether the premise is *lived* than a terse event summary. Bounded so the
    whole digest stays within the model's context even for a 50k+ word draft.
    """
    parts: list[str] = []
    for c in chapters:
        excerpt = " ".join(c.prose.split()[:words_per_chapter])
        parts.append(f"### {c.chapter_title}\n{excerpt}")
    return "\n\n".join(parts)


def _run_chapter_gates(
    world_bible: WorldBible,
    chapter_title: str,
    chapter_beats: str,
    prose: str,
    story_so_far: str,
    target_words: int,
    prior_proses: list[str],
) -> list[GateReport]:
    """Fast per-chapter gates (ambient LM, reasoning off for speed)."""
    reports: list[GateReport] = []

    # Length gate (deterministic) — under-length is blocking when a target is set.
    length = GateReport(gate="length")
    wc = _word_count(prose)
    if target_words and wc < int(target_words * 0.8):
        length.findings.append(
            EvalFinding(
                check="length",
                severity="fail",
                message=f"Chapter is {wc} words; target ~{target_words}.",
                fix=f"Expand to roughly {target_words} words by deepening the existing "
                f"scenes — more sensory detail, interiority, and consequence — not by "
                f"adding filler or repeating beats.",
                locus=chapter_title,
            )
        )
    reports.append(length)

    reports.append(bible_consistency.check(world_bible, chapter_title, prose))
    reports.append(
        comprehension.check(chapter_title, chapter_beats, world_bible, prose, story_so_far)
    )
    reports.append(phrase_freshness.check(prose, prior_proses, world_bible))
    return reports


def _combine_brief(reports: list[GateReport]) -> tuple[str, bool]:
    """Merge gate reports into one revision brief and a blocking flag."""
    briefs = [r.revision_brief() for r in reports]
    briefs = [b for b in briefs if b]
    blocking = any(r.blocking for r in reports)
    return "\n".join(briefs), blocking


class Reviewed:
    id = "reviewed"
    status = "experimental"
    description = "Self-reviewing drafter: per-chapter bible+comprehension gates with revision loop."

    def draft(self, inp: DraftingInput) -> DraftingOutput:
        # Resolve the bible against itself first so every chapter inherits one
        # consistent canon instead of each enhance-step's private version.
        internal_report, clarifications = bible_internal.check(inp.world_bible)
        canon_notes = "\n".join(f"- {c}" for c in clarifications)
        if internal_report.findings:
            update_artifact(
                inp.output_file,
                "Bible Self-Consistency",
                self._render_findings(internal_report.findings),
                level=2,
            )

        update_artifact(inp.output_file, "Final Story", "", level=2)
        chapters: list[DraftedChapter] = []
        story_so_far = ""
        continuity_ledger = ""
        total = len(inp.chapters_plan)
        target = inp.target_words_per_chapter
        max_rev = inp.max_revisions
        drafter = dspy.ChainOfThought(DraftChapterReviewed)
        ledger_updater = dspy.ChainOfThought(UpdateContinuityLedger)
        all_warnings: list[EvalFinding] = []

        for i, ch in enumerate(inp.chapters_plan, 1):
            chapter_plan_str = f"{ch.chapter_title}\n{ch.chapter_beats}"
            hint = act_hint_for_chapter(i, total, inp.spine)
            spine_for_draft = hint["spine_through_act"]

            feedback = ""
            best_prose = ""
            for attempt in range(max_rev + 1):
                result = drafter(
                    story_idea=inp.idea,
                    story_title=inp.title,
                    story_spine=spine_for_draft,
                    act_hint=hint["label"],
                    rules_of_the_world=inp.world_bible.rules_of_the_world,
                    characters=inp.world_bible.characters,
                    locations=inp.world_bible.locations,
                    timeline=inp.world_bible.timeline,
                    chapter=chapter_plan_str,
                    continuity_ledger=continuity_ledger,
                    canon_notes=canon_notes,
                    story_so_far=story_so_far,
                    target_words=target or 1200,
                    revision_feedback=feedback,
                )
                best_prose = result.prose
                reports = _run_chapter_gates(
                    inp.world_bible, ch.chapter_title, ch.chapter_beats,
                    best_prose, story_so_far, target,
                    prior_proses=[c.prose for c in chapters],
                )
                brief, blocking = _combine_brief(reports)
                wc = _word_count(best_prose)
                logger.info(
                    "ch %d/%d attempt %d: words=%d blocking=%s findings=%d",
                    i, total, attempt + 1, wc, blocking,
                    sum(len(r.findings) for r in reports),
                )
                if not blocking:
                    # accept; stash any non-blocking warnings for the report
                    for r in reports:
                        all_warnings.extend(r.warnings)
                    break
                if attempt == max_rev:
                    logger.warning(
                        "ch %d: revisions exhausted, accepting with %d blocking finding(s)",
                        i, sum(len(r.blocking) for r in reports),
                    )
                    for r in reports:
                        all_warnings.extend(r.findings)
                    break
                feedback = brief

            update_artifact(
                inp.output_file, f"Chapter {i}: {ch.chapter_title}", best_prose, level=3,
            )
            chapters.append(DraftedChapter(chapter_title=ch.chapter_title, prose=best_prose))

            plan_so_far_strs = [
                c.chapter_title + "\n" + c.chapter_beats for c in inp.chapters_plan[:i]
            ]
            story_so_far = run_generate_story_so_far(plan_so_far_strs, story_so_far, best_prose)
            try:
                continuity_ledger = ledger_updater(
                    prior_ledger=continuity_ledger,
                    chapter_title=ch.chapter_title,
                    chapter_prose=best_prose,
                ).ledger
            except Exception as exc:  # ledger is an aid, never fatal
                logger.warning("ledger update failed at ch %d: %s", i, exc)

        self._write_review_report(inp, chapters, story_so_far, all_warnings)
        return DraftingOutput(chapters=chapters, continuity_artifact=story_so_far)

    def _write_review_report(
        self,
        inp: DraftingInput,
        chapters: list[DraftedChapter],
        story_so_far: str,
        per_chapter_warnings: list[EvalFinding],
    ) -> None:
        """Run the once-per-manuscript gates (with reasoning on) and write a report."""
        total_words = sum(_word_count(c.prose) for c in chapters)
        chapter_map = {c.chapter_title: c.prose for c in chapters}

        # Per-manuscript gates run with the ambient (reasoning-off) LM: on a
        # local 27B these once-per-run gates would otherwise add tens of
        # minutes of chain-of-thought. The per-chapter gates already did the
        # close reading; this pass is the cross-chapter sweep + advisory score.
        cont = continuity.check(chapter_map)
        # Judge premise fidelity against real prose excerpts, not the terse
        # rolling summary — the summary is an event log and systematically
        # undersells how much the prose actually dramatizes the premise.
        prem = premise_fidelity.check(
            inp.idea, inp.title, inp.spine, _premise_evidence(chapters),
        )

        lines = [
            f"- **Total words:** {total_words} across {len(chapters)} chapters "
            f"(avg {total_words // max(len(chapters), 1)}/chapter)",
            "",
            "### Cross-chapter continuity",
        ]
        lines.append(self._render_findings(cont.findings))
        lines.append("")
        lines.append("### Premise fidelity")
        lines.append(self._render_findings(prem.findings))
        if per_chapter_warnings:
            lines.append("")
            lines.append("### Per-chapter warnings carried forward")
            lines.append(self._render_findings(per_chapter_warnings[:40]))

        update_artifact(inp.output_file, "Review Report", "\n".join(lines), level=2)
        logger.info(
            "review report written: total_words=%d continuity_blocking=%d",
            total_words, len(cont.blocking),
        )

    @staticmethod
    def _render_findings(findings: list[EvalFinding]) -> str:
        if not findings:
            return "_No issues found._"
        out = []
        for f in findings:
            where = f" _({f.locus})_" if f.locus else ""
            out.append(f"- **{f.severity.upper()}** [{f.check}]{where}: {f.message}")
        return "\n".join(out)


register(
    id="reviewed",
    status="experimental",
    description=Reviewed.description,
)(Reviewed)
