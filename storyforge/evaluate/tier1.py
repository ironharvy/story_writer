"""Tier 1 — deterministic checks (no LLM). Fast, free, reproducible gates."""
from __future__ import annotations

import re
from collections import Counter

from . import FAIL, PASS, WARN, Check
from .parse import Manuscript

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

# Thresholds from STORY_EVAL_SPEC.md.
LENGTH_FLOOR = 300
BAND_LOW, BAND_HIGH = 800, 2500
NAME_DRIFT_BUDGET = 2
PHRASE_REUSE_BUDGET = 5
# Pairwise 5-gram Jaccard above this reads as two chapters staging the same scene.
DUPLICATE_SCENE_JACCARD = 0.18
# Trailing characters that mark a deliberately completed sentence.
_TERMINAL_PUNCT = set(".!?")
# A final word that leaves the sentence hanging => truncation, not a stylistic ending.
_DANGLING_WORDS = {
    "and", "or", "but", "nor", "yet", "so", "because", "that", "which", "while",
    "as", "if", "then", "than", "the", "a", "an", "to", "of", "with", "for",
    "into", "onto", "from", "at", "by", "in", "on", "his", "her", "their", "its",
}

_STOP = set(
    """the a an and or but nor for so yet of to in on at by with from into onto over under
    as is are was were be been being am do does did doing have has had having will would
    shall should can could may might must not no yes if then than that this these those
    there here where when while who whom whose which what why how all any both each few more
    most other some such only own same too very just about above after again against because
    before below between down during further off out through up upon out it its it's i you he
    she they we me him her them us my your his their our mine yours hers theirs ours himself
    herself themselves myself yourself itself one two three back like said say says got get
    going go went come came see saw look looked know knew think thought time now still even
    around toward towards within without across behind beside near let made make many much
    way ways thing things something nothing anything everything someone anyone everyone""".split()
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def t1_1_length(m: Manuscript) -> Check:
    """Chapter length: 300-word hard floor (gate); 800–2500 target band (warn)."""
    short = [(c.label, c.word_count) for c in m.chapters if c.word_count < LENGTH_FLOOR]
    out_band = [(c.label, c.word_count) for c in m.chapters
                if c.word_count >= LENGTH_FLOOR and not (BAND_LOW <= c.word_count <= BAND_HIGH)]
    counts = {c.label: c.word_count for c in m.chapters}
    if not m.chapters:
        return Check("T1.1", "Chapter length", FAIL, "No chapters found.", {"counts": counts})
    if short:
        return Check("T1.1", "Chapter length", FAIL,
                     f"{len(short)} chapter(s) below the {LENGTH_FLOOR}-word floor: "
                     + ", ".join(f"{l} ({w}w)" for l, w in short),
                     {"counts": counts, "below_floor": short})
    if out_band:
        return Check("T1.1", "Chapter length", WARN,
                     f"{len(out_band)} chapter(s) outside the {BAND_LOW}-{BAND_HIGH}w band: "
                     + ", ".join(f"{l} ({w}w)" for l, w in out_band),
                     {"counts": counts, "out_of_band": out_band})
    return Check("T1.1", "Chapter length", PASS,
                 f"All {len(m.chapters)} chapters within the {BAND_LOW}-{BAND_HIGH}w band.",
                 {"counts": counts})


def t1_2_presence(m: Manuscript) -> Check:
    """Every canonical character appears in the prose (full name or first token)."""
    prose = m.prose.lower()
    missing = []
    for ch in m.characters:
        full = re.escape(ch.canonical.lower())
        first = re.escape(ch.first_token.lower())
        if re.search(rf"\b{full}\b", prose) or re.search(rf"\b{first}\b", prose):
            continue
        missing.append(ch.canonical)
    if not m.characters:
        return Check("T1.2", "Character presence", WARN, "No cast entries parsed.")
    if missing:
        return Check("T1.2", "Character presence", FAIL,
                     f"{len(missing)} cast member(s) never appear in the prose: "
                     + ", ".join(missing), {"missing": missing})
    return Check("T1.2", "Character presence", PASS,
                 f"All {len(m.characters)} cast members appear in the prose.")


def t1_3_name_drift(m: Manuscript) -> Check:
    """Variant spellings of canonical names (same first token diverging, or near-miss)."""
    canon_first = {c.first_token for c in m.characters}
    canon_full = {c.canonical for c in m.characters}
    canon_first_lower = {c.lower(): c for c in canon_first}
    flags: list[dict] = []
    seen = set()
    generic = {"The", "A", "An", "And", "But", "She", "He", "They", "It", "Then", "When",
               "Her", "His", "Their", "I", "We", "You"}
    for pn in set(_PROPER.findall(m.prose)):
        if pn in canon_full or pn in canon_first:
            continue
        first = pn.split()[0]
        if first in generic:
            continue
        flagged_as = None
        # (a) same first token, divergent full form, not a clean prefix either way
        if first in canon_first:
            cands = [c for c in canon_full if c.split()[0] == first]
            if not any(c.startswith(pn) or pn.startswith(c) for c in cands):
                flagged_as = cands[0] if cands else first
        # (b) single-token near miss of a canonical first token (Cinder/Cindar)
        elif len(pn.split()) == 1 and len(first) >= 4:
            for cf_lower, cf in canon_first_lower.items():
                d = _levenshtein(first.lower(), cf_lower)
                if 0 < d <= 2 and not cf_lower.startswith(first.lower()) \
                        and not first.lower().startswith(cf_lower):
                    flagged_as = cf
                    break
        if flagged_as and pn.lower() not in seen:
            seen.add(pn.lower())
            flags.append({"variant": pn, "canonical": flagged_as})
    if flags:
        sev = WARN
        detail = ", ".join(f"{f['variant']}~{f['canonical']}" for f in flags)
        return Check("T1.3", "Name drift", sev,
                     f"{len(flags)} possible name variant(s): {detail}"
                     + (f" (budget {NAME_DRIFT_BUDGET})" if len(flags) > NAME_DRIFT_BUDGET else ""),
                     {"flags": flags, "budget": NAME_DRIFT_BUDGET})
    return Check("T1.3", "Name drift", PASS, "No name drift detected.")


def _ngrams(words: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def t1_4_phrase_reuse(m: Manuscript) -> Check:
    """5-grams shared across >=2 chapters (excluding world/premise context)."""
    context = _ngrams(_words(m.world_text))
    per_chapter = [_ngrams(_words(c.prose)) for c in m.chapters]
    counter: Counter = Counter()
    for grams in per_chapter:
        for g in grams - context:
            counter[g] += 1
    shared = [g for g, cnt in counter.items() if cnt >= 2]
    n = len(shared)
    examples = [" ".join(g) for g in sorted(shared)[:8]]
    if n > PHRASE_REUSE_BUDGET:
        return Check("T1.4", "Cross-chapter phrase reuse", WARN,
                     f"{n} distinct 5-grams reused across chapters (budget {PHRASE_REUSE_BUDGET}). "
                     f"e.g.: {'; '.join(examples[:5])}",
                     {"count": n, "budget": PHRASE_REUSE_BUDGET, "examples": examples})
    if n:
        return Check("T1.4", "Cross-chapter phrase reuse", PASS,
                     f"{n} reused 5-gram(s), within budget {PHRASE_REUSE_BUDGET}.",
                     {"count": n, "budget": PHRASE_REUSE_BUDGET, "examples": examples})
    return Check("T1.4", "Cross-chapter phrase reuse", PASS, "No cross-chapter 5-gram reuse.")


def t1_5_over_repetition(m: Manuscript) -> Check:
    """Content words repeated absurdly often relative to manuscript length."""
    names = {c.first_token.lower() for c in m.characters}
    names |= {t.lower() for c in m.characters for t in c.canonical.split()}

    def _depossess(w: str) -> str:
        for suf in ("'s", "’s"):
            if w.endswith(suf):
                return w[: -len(suf)]
        return w.rstrip("'’")

    words = [_depossess(w) for w in _words(m.prose)]
    content = [w for w in words if len(w) >= 4 and w not in _STOP and w not in names]
    wc = len(_words(m.prose))
    if wc == 0:
        return Check("T1.5", "Over-repetition", WARN, "No prose to analyze.")
    freq = Counter(content)
    # Exempt the story's legitimate core vocabulary (the "~20 most-frequent words"),
    # scaled down for short texts so the check stays sensitive on small fixtures.
    exempt_n = min(20, len(freq) // 3)
    exempt = {w for w, _ in freq.most_common(exempt_n)}
    threshold = max(8, wc // 400)
    flagged = [(w, c) for w, c in freq.items() if w not in exempt and c > threshold]
    flagged.sort(key=lambda x: -x[1])
    if flagged:
        return Check("T1.5", "Over-repetition", WARN,
                     f"{len(flagged)} content word(s) over-repeated (>{threshold}x): "
                     + ", ".join(f"{w}×{c}" for w, c in flagged[:8]),
                     {"threshold": threshold, "flagged": flagged[:20]})
    return Check("T1.5", "Over-repetition", PASS,
                 f"No content word exceeds {threshold} occurrences (outside top-20).")


def t1_6_duplicate_scene(m: Manuscript) -> Check:
    """Near-duplicate chapters: pairwise 5-gram Jaccard overlap (world/premise
    n-grams excluded). WARN on any pair above threshold — the deterministic
    signature of a scene staged twice (e.g. a climax written in two chapters)."""
    context = _ngrams(_words(m.world_text))
    per = [_ngrams(_words(c.prose)) - context for c in m.chapters]
    pairs: list[dict] = []
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            a, b = per[i], per[j]
            union = len(a | b)
            if not union:
                continue
            jac = len(a & b) / union
            if jac >= DUPLICATE_SCENE_JACCARD:
                pairs.append({"a": m.chapters[i].label, "b": m.chapters[j].label,
                              "jaccard": round(jac, 3)})
    if pairs:
        pairs.sort(key=lambda p: -p["jaccard"])
        detail = ", ".join(f"{p['a']}~{p['b']} ({p['jaccard']})" for p in pairs[:5])
        return Check("T1.6", "Duplicate scene", WARN,
                     f"{len(pairs)} chapter pair(s) overlap heavily (5-gram Jaccard >= "
                     f"{DUPLICATE_SCENE_JACCARD}): {detail}",
                     {"threshold": DUPLICATE_SCENE_JACCARD, "pairs": pairs})
    return Check("T1.6", "Duplicate scene", PASS,
                 f"No chapter pair exceeds {DUPLICATE_SCENE_JACCARD} 5-gram overlap.",
                 {"threshold": DUPLICATE_SCENE_JACCARD})


def t1_7_ending_completeness(m: Manuscript) -> Check:
    """Final chapter must actually END — not get cut off mid-thought. Deterministic
    (immune to judge hallucination): FAIL true truncation (dangling conjunction or
    comma / no terminal punctuation); WARN stylistic dash/ellipsis. Absolute length
    is left to T1.1 (which already gates every chapter's 300-word floor)."""
    if not m.chapters:
        return Check("T1.7", "Ending completeness", WARN, "No chapters to check.")
    last = m.chapters[-1]
    text = last.prose.strip()
    wc = last.word_count
    details = {"final_chapter": last.label, "word_count": wc}
    if not text:
        return Check("T1.7", "Ending completeness", FAIL,
                     f"Final chapter ({last.label}) has no prose.", details)
    # Strip trailing closing quotes/brackets to find the real sentence terminator.
    stripped = text.rstrip("\"'”’) ")
    last_char = stripped[-1] if stripped else ""
    words = _WORD.findall(stripped.lower())
    last_word = words[-1] if words else ""
    details["tail"] = text[-60:]
    if last_char in _TERMINAL_PUNCT:
        return Check("T1.7", "Ending completeness", PASS,
                     f"Final chapter ends on a complete sentence ('…{stripped[-24:]}').",
                     details)
    if last_char in "—–" or stripped.endswith("...") or last_char == "…":
        return Check("T1.7", "Ending completeness", WARN,
                     f"Final chapter ends on a dash/ellipsis ('…{text[-24:]}') — "
                     "stylistic; verify it is intentional.", details)
    if last_char == "," or last_word in _DANGLING_WORDS:
        return Check("T1.7", "Ending completeness", FAIL,
                     f"Final chapter ends mid-thought with no terminal punctuation "
                     f"('…{text[-40:]}') — the ending appears truncated.", details)
    return Check("T1.7", "Ending completeness", WARN,
                 f"Final chapter does not end with terminal punctuation "
                 f"('…{text[-40:]}').", details)


def run_tier1(m: Manuscript) -> list[Check]:
    return [t1_1_length(m), t1_2_presence(m), t1_3_name_drift(m),
            t1_4_phrase_reuse(m), t1_5_over_repetition(m),
            t1_6_duplicate_scene(m), t1_7_ending_completeness(m)]
