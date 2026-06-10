"""Phrase-freshness gate (deterministic).

The one prose-quality weakness the cross-chapter QA heuristic reliably surfaces
on long local-model drafts is *stock-phrase reuse*: the model leans on the same
crutch phrasings chapter after chapter ("his heart hammered against his ribs",
"the weight of his hoard", "looked at his hands"). ``qa.check_cross_chapter_
phrase_reuse`` reports this across a finished manuscript, but by then it's baked
in. This gate runs the same n-gram comparison *incrementally* — the chapter just
drafted vs. everything before it — so the reviewed generator can hand the
offending phrases back as a "vary these" revision brief while the chapter is
still cheap to change.

Pure/deterministic: no model call, fully unit-testable. Character names are
excluded so a recurring protagonist name isn't mistaken for a stock phrase.
"""
from __future__ import annotations

import logging
import math

import qa
from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)

DEFAULT_N = 5
# A "crutch" phrase is one the model leans on PERVASIVELY — recurring in a
# meaningful fraction of the chapters written so far, not just any 5-gram that
# happens to appear twice. Using a fraction (rather than an absolute chapter
# count) keeps the crutch set — and therefore the finding counts and thresholds
# below — stable whether the novel is 5 chapters or 40: late chapters in a long
# book aren't penalised just because a large prior corpus shares incidental
# n-grams. Floor of 2 so the gate still works on short drafts.
CRUTCH_FRACTION = 0.25
MIN_CRUTCH_CHAPTERS = 2
WARN_THRESHOLD = 4
# Block only egregiously stale chapters so the revision loop isn't dominated by
# phrase-chasing; lighter reuse rides along as a WARN folded into other fixes.
BLOCK_THRESHOLD = 12


def _crutch_min_chapters(n_prior: int) -> int:
    """How many prior chapters a phrase must recur in to count as a crutch,
    scaled to the corpus so the threshold is corpus-size stable."""
    return max(MIN_CRUTCH_CHAPTERS, math.ceil(CRUTCH_FRACTION * n_prior))


def _name_tokens(world_bible) -> set[str]:
    """Lowercased single tokens of canonical character names, to exclude from
    reuse detection (a recurring name is not a stock phrase)."""
    toks: set[str] = set()
    for bullet in getattr(world_bible, "characters", []) or []:
        for word in qa.canonical_name(bullet).split():
            w = word.strip(".,:;—–-").lower()
            if w:
                toks.add(w)
    return toks


def check(prose: str, prior_chapters: list[str], world_bible=None,
          n: int = DEFAULT_N) -> GateReport:
    """Flag n-gram phrases in ``prose`` that already appeared in ``prior_chapters``.

    Returns at most one finding summarising the worst offenders, with a revision
    brief naming the phrases to vary. Empty report for the first chapter.
    """
    report = GateReport(gate="phrase_freshness")
    if not prior_chapters:
        return report

    names = _name_tokens(world_bible) if world_bible is not None else set()

    # Count, per n-gram, how many distinct prior chapters it appeared in. Only
    # grams recurring in >= MIN_PRIOR_CHAPTERS are treated as stock crutches.
    prior_chapter_count: dict[tuple, int] = {}
    for chap in prior_chapters:
        for gram in qa._ngrams(qa._words(chap), n):
            prior_chapter_count[gram] = prior_chapter_count.get(gram, 0) + 1
    min_chapters = _crutch_min_chapters(len(prior_chapters))
    crutches = {g for g, c in prior_chapter_count.items() if c >= min_chapters}

    cur_grams = qa._ngrams(qa._words(prose), n)

    overlap = []
    for gram in cur_grams & crutches:
        # ignore grams that are mostly a character name
        content = [t for t in gram if t not in names]
        if len(content) < n - 1:
            continue
        overlap.append(" ".join(gram))

    count = len(overlap)
    if count < WARN_THRESHOLD:
        return report

    overlap.sort()
    shown = overlap[:12]
    severity = "fail" if count >= BLOCK_THRESHOLD else "warn"
    report.findings.append(
        EvalFinding(
            check="phrase_freshness",
            severity=severity,
            message=f"{count} phrase(s) reused verbatim from earlier chapters",
            fix="Rewrite these repeated phrasings with fresh language (vary verbs, "
            "imagery, and sentence shape; don't reuse the same stock beats): "
            + "; ".join(f'"{p}"' for p in shown),
        )
    )
    logger.info("phrase_freshness: %d reused %d-gram(s) (%s)", count, n, severity)
    return report
