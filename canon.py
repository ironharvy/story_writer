"""The shared canon/continuity config — one source of truth for the rules that
span chapters.

A project's canon lives in a machine-readable file (``canon/<project>.yaml`` or
``.json``) and is consumed two ways:

* **Generation (prevent):** :func:`render_chapter_slice` turns the slice of the
  canon relevant to one chapter into a plain-text directive block that is fed
  into the per-chapter drafting signatures (names allowed this chapter, the
  locked appearance, the POV rule, "no scaffolding vocabulary", and any required
  artifact for that chapter).
* **Evaluation (score/gate):** the deterministic metrics in
  :mod:`canon_metrics` and the LM judges in :mod:`bench.judge` read the same
  :class:`Canon` object.

The :class:`Canon` schema is intentionally generic/per-project — the pipeline
writes other stories from other ideas — so only the *values* in the YAML are
story-specific. See ``docs/canon-schema.md`` for the schema contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default location used when no explicit path is given. A project ships exactly
# one canon file; this is the "Nameless Weapon" instance.
DEFAULT_CANON_PATH = Path(__file__).resolve().parent / "canon" / "nameless_weapon.yaml"


@dataclass(frozen=True)
class Canon:
    """A parsed canon/continuity config.

    Every field defaults to empty so a partial canon (or no canon at all) is
    valid: a project that does not pin, say, ``required_artifacts`` simply has
    none enforced. ``raw`` keeps the untouched mapping for forward-compat.
    """

    project: str = ""
    title: str = ""
    narration: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    name_arc: dict[str, Any] = field(default_factory=dict)
    locked_appearance: dict[str, Any] = field(default_factory=dict)
    required_artifacts: dict[int, dict] = field(default_factory=dict)
    canon_invariants: list[dict] = field(default_factory=list)
    accepted_aliases: dict[str, str] = field(default_factory=dict)
    motif_allowlist: list[str] = field(default_factory=list)
    scaffolding_blocklist: list[str] = field(default_factory=list)
    device_allowlist: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # --- name-arc helpers ----------------------------------------------------

    def names_not_yet_allowed(self, chapter: int) -> list[str]:
        """Names whose earliest-narration chapter is strictly after ``chapter``."""
        earliest = self.name_arc.get("earliest_narration", {})
        return sorted(n for n, ch in earliest.items() if chapter < int(ch))

    def names_allowed(self, chapter: int) -> list[str]:
        """Restricted names that have become available by ``chapter``."""
        earliest = self.name_arc.get("earliest_narration", {})
        return sorted(n for n, ch in earliest.items() if chapter >= int(ch))

    def required_artifact(self, chapter: int) -> dict | None:
        return self.required_artifacts.get(int(chapter))

    def active_invariants(self, chapter: int) -> list[dict]:
        """Invariants in force as of ``chapter`` (those with no ``after_chapter``
        or whose ``after_chapter`` is < ``chapter``)."""
        out: list[dict] = []
        for inv in self.canon_invariants:
            after = inv.get("after_chapter")
            if after is None or chapter > int(after):
                out.append(inv)
        return out


def _coerce_int_keys(mapping: Any) -> dict[int, Any]:
    """Coerce a mapping's keys to ``int`` where possible (YAML/JSON object keys
    for chapter numbers can arrive as ``str``)."""
    if not isinstance(mapping, dict):
        return {}
    out: dict[int, Any] = {}
    for key, value in mapping.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


def from_dict(data: dict) -> Canon:
    """Build a :class:`Canon` from an already-parsed mapping."""
    data = data or {}
    name_arc = dict(data.get("name_arc", {}) or {})
    if "call_order" in name_arc:
        name_arc["call_order"] = _coerce_int_keys(name_arc["call_order"])
    return Canon(
        project=str(data.get("project", "")),
        title=str(data.get("title", "")),
        narration=dict(data.get("narration", {}) or {}),
        timeline=dict(data.get("timeline", {}) or {}),
        name_arc=name_arc,
        locked_appearance=dict(data.get("locked_appearance", {}) or {}),
        required_artifacts=_coerce_int_keys(data.get("required_artifacts", {})),
        canon_invariants=list(data.get("canon_invariants", []) or []),
        accepted_aliases=dict(data.get("accepted_aliases", {}) or {}),
        motif_allowlist=list(data.get("motif_allowlist", []) or []),
        scaffolding_blocklist=list(data.get("scaffolding_blocklist", []) or []),
        device_allowlist=list(data.get("device_allowlist", []) or []),
        raw=data,
    )


def load_canon(path: str | Path | None = None) -> Canon:
    """Load a canon config from YAML or JSON.

    ``path=None`` loads :data:`DEFAULT_CANON_PATH` if present, else returns an
    empty :class:`Canon` (so the pipeline runs canon-free for projects that
    haven't authored one yet).
    """
    if path is None:
        if not DEFAULT_CANON_PATH.exists():
            return Canon()
        path = DEFAULT_CANON_PATH
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    return from_dict(data or {})


def apply_accepted_aliases(text: str, canon: Canon) -> str:
    """Rewrite each accepted spoken/spelled-out alias to its canonical form.

    Used before name-drift detection so a sanctioned form ("Vessel Eighty-Four")
    is not mistaken for drift from its canonical ("Vessel 84"). Longer aliases are
    applied first so a contained alias ("Eighty-Four") doesn't pre-empt the longer
    match."""
    for spoken in sorted(canon.accepted_aliases, key=len, reverse=True):
        canonical = canon.accepted_aliases[spoken]
        text = re.sub(re.escape(spoken), canonical, text, flags=re.IGNORECASE)
    return text


# --- generation-time rendering ----------------------------------------------


def _appearance_lines(locked: dict) -> list[str]:
    """Render the protagonist's locked physical anchor as instruction lines."""
    proto = locked.get("protagonist", {}) if locked else {}
    if not proto:
        return []
    traits: list[str] = []
    for key in ("eyes", "hair", "hair_style", "build"):
        if proto.get(key):
            label = key.replace("_", " ")
            traits.append(f"{label}: {proto[key]}")
    marks = proto.get("permanent_marks") or []
    if marks:
        traits.append("permanent mark(s): " + ", ".join(marks))
    if not traits:
        return []
    lines = [
        "GROUND THE POV CHARACTER PHYSICALLY. The point-of-view character has a "
        "fixed appearance — anchor them in the prose with these exact traits, and "
        "never contradict them:",
    ]
    lines += [f"  - {t}" for t in traits]
    principle = proto.get("principle")
    if principle:
        lines.append(f"  - principle: {principle}")
    return lines


def render_chapter_slice(canon: Canon, chapter: int) -> str:
    """Render the canon directives relevant to ``chapter`` as a plain-text block.

    This is what gets injected into the per-chapter drafting signatures. An empty
    canon renders to an empty string (no-op injection).
    """
    if canon == Canon():
        return ""

    sections: list[str] = [f"CANON DIRECTIVES FOR CHAPTER {chapter}:"]

    pov_rule = (canon.narration or {}).get("pov_rule")
    if pov_rule:
        sections.append(f"POINT OF VIEW: {pov_rule}")

    sections += _appearance_lines(canon.locked_appearance) or []

    not_yet = canon.names_not_yet_allowed(chapter)
    if not_yet:
        sections.append(
            "NAMES NOT YET REVEALED — do not use these names in narration yet: "
            + ", ".join(not_yet)
            + "."
        )

    artifact = canon.required_artifact(chapter)
    if artifact:
        sections.append(
            f"REQUIRED THIS CHAPTER: this chapter must include the "
            f"{artifact.get('name', 'required artifact')}."
        )

    invariants = canon.active_invariants(chapter)
    if invariants:
        sections.append("CANON INVARIANTS in force:")
        sections += [f"  - {inv.get('rule', '')}" for inv in invariants]

    if canon.scaffolding_blocklist:
        sections.append(
            "NO SCAFFOLDING VOCABULARY: this is finished prose. Never write "
            "planning/production language such as chapter numbers ('Ch. 4'), "
            "'Beat 3', 'world bible', 'setup/payoff', 'easter egg', 'POV', or "
            "address 'the reader'."
        )

    return "\n".join(sections)
