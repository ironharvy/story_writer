"""The generation pipeline: idea -> premise -> bible -> spine -> drafted chapters.

Outline-first ("Snowflake") with scene/sequel chapter plans and a bounded
critic->revise pass per chapter. Every artifact is written to disk the moment it
is produced, and `run(resume=True)` continues an interrupted run from its dir.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config, llm, prompts
from .models import (Chapter, ChapterPlan, Character, Location, Premise, RunState,
                     StorySpec, WorldBible)

log = logging.getLogger("storyforge.pipeline")

_HEADING_LINE = re.compile(r"^\s*(#{1,6}\s+.*|chapter\s+\w+.*|\*\*.*\*\*)\s*$", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")


def _wc(text: str) -> int:
    return len(_WORD.findall(text))


def _as_dict(d: object) -> dict:
    """Models sometimes emit a bare list/scalar where an object was asked for; the
    caller can still proceed (with defaults) instead of crashing on `.get`."""
    return d if isinstance(d, dict) else {}


def _s(d: dict, k: str, default: str = "") -> str:
    v = d.get(k, default)
    return v if isinstance(v, str) else (str(v) if v not in (None, {}, []) else default)


def _slist(d: dict, k: str) -> list[str]:
    v = d.get(k, [])
    if isinstance(v, list):
        return [x if isinstance(x, str) else _s({"x": x}, "x") for x in v]
    return [v] if isinstance(v, str) and v else []


def clean_prose(text: str) -> str:
    """Strip a leading heading / 'Chapter N' label and surrounding fences the model
    may add despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    lines = text.splitlines()
    while lines and (_HEADING_LINE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


class Pipeline:
    def __init__(self, draft_model: str, run_dir: Path, cfg: config.RunConfig | None = None,
                 use_critic: bool = True, use_derepeat: bool = True, console=None,
                 climax_model: str = "", scene_level: bool = False):
        self.model = draft_model
        self.cfg = cfg or config.RunConfig(draft_model=draft_model, judge_model=draft_model,
                                           temperature=0.8)
        self.run_dir = Path(run_dir)
        self.use_critic = use_critic
        self.use_derepeat = use_derepeat
        self.console = console
        # Optional escalation: a stronger model used ONLY for the climax chapter,
        # bounding cost to one chapter while lifting the most important scene.
        self.climax_model = climax_model
        self.scene_level = scene_level
        (self.run_dir / "chapters").mkdir(parents=True, exist_ok=True)

    def _chapter_model(self, plan: ChapterPlan) -> str:
        """The model to draft a given chapter — escalate the climax if configured."""
        if self.climax_model and (plan.structural_role or "").lower() in ("climax", "resolution"):
            return self.climax_model
        return self.model

    # --- io ---
    def _say(self, msg: str) -> None:
        if self.console:
            self.console.print(msg)
        else:
            print(msg)
        log.info(msg)

    def _save(self, state: RunState) -> None:
        (self.run_dir / "state.json").write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False))

    # --- planning gen calls (low temperature, JSON) ---
    def gen_spec(self, idea: str, overrides: dict) -> StorySpec:
        provided = {k: v for k, v in overrides.items() if v}
        d = _as_dict(llm.complete_json(prompts.infer_spec_prompt(idea, provided), system=prompts.GEN_SYSTEM,
                              model=self.model, cfg=self.cfg, temperature=0.3, trace_name="spec"))
        spec = StorySpec()
        for k in ("genre", "tone", "pov", "ending", "audience"):
            if d.get(k):
                setattr(spec, k, _s(d, k))
        for k, v in overrides.items():
            if v:
                setattr(spec, k, v)
        if spec.pov not in ("first_person", "third_person_limited", "third_person_omniscient"):
            spec.pov = "third_person_limited"
        return spec

    def gen_premise(self, idea: str, spec: StorySpec) -> Premise:
        d = _as_dict(llm.complete_json(prompts.premise_prompt(idea, spec), system=prompts.GEN_SYSTEM,
                              model=self.model, cfg=self.cfg, temperature=0.7, trace_name="premise"))
        return Premise(title=_s(d, "title", "Untitled"), logline=_s(d, "logline"),
                       paragraph=_s(d, "paragraph"), theme=_s(d, "theme"))

    def gen_bible(self, idea: str, spec: StorySpec, premise: Premise) -> WorldBible:
        d = llm.complete_json(prompts.bible_prompt(idea, spec, premise), system=prompts.GEN_SYSTEM,
                              model=self.model, cfg=self.cfg, temperature=0.6, trace_name="bible")
        # The model occasionally returns a bare list of characters instead of the
        # {characters, locations, rules} object — tolerate both (cf. gen_spine).
        if isinstance(d, list):
            d = {"characters": d}
        elif not isinstance(d, dict):
            d = {}
        chars = []
        for c in (d.get("characters") or []):
            if not isinstance(c, dict) or not c.get("name"):
                continue
            chars.append(Character(name=_s(c, "name"), role=_s(c, "role"),
                                   description=_s(c, "description"), want=_s(c, "want"),
                                   need=_s(c, "need"), arc=_s(c, "arc")))
        if not chars:
            raise llm.LLMError("bible produced no characters")
        if not any(c.role == "protagonist" for c in chars):
            chars[0].role = "protagonist"
        locs = [Location(name=_s(l, "name"), description=_s(l, "description"))
                for l in (d.get("locations") or []) if isinstance(l, dict) and l.get("name")]
        return WorldBible(characters=chars, locations=locs, rules=_slist(d, "rules"))

    def gen_spine(self, spec: StorySpec, premise: Premise, bible: WorldBible) -> list[ChapterPlan]:
        protagonist = next((c.name for c in bible.characters if c.role == "protagonist"),
                           bible.names()[0])
        d = llm.complete_json(prompts.spine_prompt(spec, premise, bible), system=prompts.GEN_SYSTEM,
                              model=self.model, cfg=self.cfg, temperature=0.6, max_tokens=4096,
                              trace_name="spine")
        rows = d.get("chapters") if isinstance(d, dict) else d
        plans: list[ChapterPlan] = []
        for i, c in enumerate(rows or [], 1):
            if not isinstance(c, dict):
                continue
            try:
                num = int(c.get("number", i))
            except (TypeError, ValueError):
                num = i
            try:
                wt = int(c.get("word_target", spec.words_per_chapter))
            except (TypeError, ValueError):
                wt = spec.words_per_chapter
            plans.append(ChapterPlan(
                number=num, title=_s(c, "title", f"Chapter {num}"),
                pov_character=protagonist,  # enforce single POV character
                summary=_s(c, "summary"), beats=_slist(c, "beats"),
                goal=_s(c, "goal"), conflict=_s(c, "conflict"), outcome=_s(c, "outcome"),
                characters_present=_slist(c, "characters_present") or [protagonist],
                structural_role=_s(c, "structural_role").lower().strip(),
                word_target=max(900, wt)))
        plans.sort(key=lambda p: p.number)
        for i, p in enumerate(plans, 1):
            p.number = i
        # Guarantee exactly one climax: if the model assigned none (or several),
        # the final chapter is the climax+resolution per the spine prompt.
        if sum(1 for p in plans if p.structural_role == "climax") != 1 and plans:
            for p in plans:
                if p.structural_role == "climax":
                    p.structural_role = "rising"
            plans[-1].structural_role = "climax"
        return plans

    # --- drafting ---
    def draft_chapter(self, spec: StorySpec, premise: Premise, bible: WorldBible,
                      plan: ChapterPlan, synopsis: str, prev_tail: str,
                      avoid: list[str], next_plan: ChapterPlan | None = None) -> str:
        model = self._chapter_model(plan)
        prompt = prompts.draft_chapter_prompt(spec, premise, bible, plan, synopsis,
                                              prev_tail, avoid, next_plan)
        prose = clean_prose(llm.complete(prompt, system=prompts.GEN_SYSTEM, model=model,
                                         cfg=self.cfg, temperature=0.85, max_tokens=8192,
                                         trace_name=f"draft-ch{plan.number}"))
        # Word-floor guard: expand if the model came in thin.
        floor = max(800, int(plan.word_target * 0.6))
        for _ in range(2):
            if _wc(prose) >= floor:
                break
            self._say(f"    chapter {plan.number} thin ({_wc(prose)}w); expanding…")
            prose = clean_prose(llm.complete(
                prompt + f"\n\nYour draft was too short. Expand it to about "
                f"{plan.word_target} words by deepening scenes (more action, dialogue, "
                "sensory detail) — do not summarize. Output ONLY the prose.",
                system=prompts.GEN_SYSTEM, model=model, cfg=self.cfg, temperature=0.85,
                max_tokens=8192, trace_name=f"draft-ch{plan.number}-expand"))
        if self.use_critic:
            prose = self._critic_pass(spec, plan, synopsis, prose)
        return prose

    def draft_chapter_scenes(self, spec: StorySpec, premise: Premise, bible: WorldBible,
                             plan: ChapterPlan, synopsis: str, prev_tail: str,
                             avoid: list[str], next_plan: ChapterPlan | None = None) -> str:
        """Scene-level redraft: write each beat as its own dramatized scene, then
        stitch. Lifts scene_vs_summary / character_agency at the cost of more LLM
        calls — gated behind --scene-level / auto-triggered on a persistently low axis."""
        beats = [b for b in (plan.beats or []) if b.strip()] or [plan.summary or plan.title]
        model = self._chapter_model(plan)
        scenes: list[str] = []
        running = prev_tail
        for i, beat in enumerate(beats, 1):
            try:
                prose = clean_prose(llm.complete(
                    prompts.scene_prompt(spec, premise, bible, plan, synopsis, running,
                                         avoid, beat, i, len(beats), next_plan),
                    system=prompts.GEN_SYSTEM, model=model, cfg=self.cfg, temperature=0.85,
                    max_tokens=4096, trace_name=f"scene-ch{plan.number}-{i}"))
            except llm.LLMError:
                continue
            if prose:
                scenes.append(prose)
                running = " ".join(prose.split()[-120:])
        stitched = "\n\n".join(scenes).strip()
        if not stitched:
            # Fall back to the single-shot drafter if scene assembly produced nothing.
            return self.draft_chapter(spec, premise, bible, plan, synopsis, prev_tail,
                                      avoid, next_plan)
        if self.use_critic:
            stitched = self._critic_pass(spec, plan, synopsis, stitched)
        return stitched

    def _critic_pass(self, spec: StorySpec, plan: ChapterPlan, synopsis: str, prose: str) -> str:
        try:
            d = llm.complete_json(prompts.critique_prompt(spec, plan, synopsis, prose),
                                  system=prompts.GEN_SYSTEM, model=self.model, cfg=self.cfg,
                                  temperature=0.2, trace_name=f"critique-ch{plan.number}")
        except llm.LLMError:
            return prose
        if d.get("ok", True) or not d.get("fixes"):
            return prose
        fixes = _slist(d, "fixes")[:5]
        self._say(f"    chapter {plan.number} revise: {'; '.join(fixes)[:120]}")
        try:
            revised = clean_prose(llm.complete(
                prompts.revise_prompt(plan, prose, fixes), system=prompts.GEN_SYSTEM,
                model=self.model, cfg=self.cfg, temperature=0.7, max_tokens=8192,
                trace_name=f"revise-ch{plan.number}"))
            if _wc(revised) >= max(700, int(plan.word_target * 0.5)):
                return revised
        except llm.LLMError:
            pass
        return prose

    def summarize_chapter(self, plan: ChapterPlan, prose: str) -> str:
        try:
            d = llm.complete_json(prompts.summary_prompt(plan, prose), system=prompts.GEN_SYSTEM,
                                  model=self.model, cfg=self.cfg, temperature=0.2,
                                  trace_name=f"summary-ch{plan.number}")
            facts = _slist(d, "facts")
            summary = _s(d, "summary")
            return summary + (("  Facts: " + "; ".join(facts)) if facts else "")
        except llm.LLMError:
            return plan.summary

    def derepeat(self, state: RunState, rounds: int = 2) -> None:
        """Final polish: rewrite later-occurring chapters to remove cross-chapter
        phrase reuse until within the T1.4 budget. Targets the metric directly."""
        from collections import Counter, defaultdict

        from .evaluate.parse import parse_manuscript
        from .evaluate.tier1 import PHRASE_REUSE_BUDGET, _ngrams, _words

        for r in range(rounds):
            chaps = sorted(state.chapters, key=lambda c: c.number)
            if len(chaps) < 2:
                return
            context = _ngrams(_words(parse_manuscript(self.assemble(state)).world_text))
            per = [_ngrams(_words(c.prose)) - context for c in chaps]
            counts: Counter = Counter()
            for grams in per:
                for g in grams:
                    counts[g] += 1
            repeated = [g for g, c in counts.items() if c >= 2]
            if len(repeated) <= PHRASE_REUSE_BUDGET:
                return
            self._say(f"• De-repetition pass {r + 1}: {len(repeated)} reused phrases "
                      f"(> {PHRASE_REUSE_BUDGET}); rewriting duplicates…")
            to_fix: dict[int, set] = defaultdict(set)
            for g in repeated:
                idxs = [i for i, grams in enumerate(per) if g in grams]
                for i in idxs[1:]:  # keep the earliest occurrence, fix the later ones
                    to_fix[i].add(" ".join(g))
            for i, phrases in sorted(to_fix.items()):
                ch = chaps[i]
                try:
                    revised = clean_prose(llm.complete(
                        prompts.derepeat_prompt(ch.number, ch.prose, list(phrases)[:20]),
                        system=prompts.GEN_SYSTEM, model=self.model, cfg=self.cfg,
                        temperature=0.8, max_tokens=8192, trace_name=f"derepeat-ch{ch.number}"))
                except llm.LLMError:
                    continue
                if _wc(revised) >= max(700, int(_wc(ch.prose) * 0.7)):
                    ch.prose = revised
                    (self.run_dir / "chapters" / f"ch{ch.number:02d}.md").write_text(
                        f"### Chapter {ch.number}: {ch.title}\n\n{revised}\n")
            self._save(state)

    def polish(self, state: RunState, notes: list[str],
               climax_numbers: set[int] | None = None) -> Path:
        """Rubric-driven revision pass: re-edit each chapter against editorial notes,
        then re-run de-repetition and reassemble. Embodies self-evaluate -> revise."""
        climax_numbers = climax_numbers or set()
        spec, premise, bible = state.spec, state.premise, state.bible
        plan_by_num = {p.number: p for p in state.spine}
        chaps = sorted(state.chapters, key=lambda c: c.number)
        for ch in chaps:
            plan = plan_by_num.get(ch.number) or ChapterPlan(
                number=ch.number, title=ch.title, pov_character=spec.pov_character)
            synopsis = "\n".join(f"Ch{c.number} ({c.title}): {c.summary}"
                                 for c in chaps if c.summary and c.number != ch.number)
            self._say(f"• Polishing chapter {ch.number}: {ch.title}")
            try:
                revised = clean_prose(llm.complete(
                    prompts.polish_chapter_prompt(spec, premise, bible, plan, ch.prose,
                                                  synopsis, notes, ch.number in climax_numbers),
                    system=prompts.GEN_SYSTEM, model=self._chapter_model(plan), cfg=self.cfg,
                    temperature=0.75, max_tokens=8192, trace_name=f"polish-ch{ch.number}"))
            except llm.LLMError:
                continue
            if _wc(revised) >= max(800, int(_wc(ch.prose) * 0.7)):
                ch.prose = revised
                (self.run_dir / "chapters" / f"ch{ch.number:02d}.md").write_text(
                    f"### Chapter {ch.number}: {ch.title}\n\n{revised}\n")
                self._say(f"    ✓ {_wc(revised)} words")
            self._save(state)
        if self.use_derepeat:
            self.derepeat(state, rounds=3)
        manuscript = self.assemble(state)
        (self.run_dir / "manuscript.md").write_text(manuscript)
        self._say(f"✓ Polished manuscript ({_wc(manuscript)} words)")
        return self.run_dir / "manuscript.md"

    def redraft_scene_level(self, state: RunState) -> Path:
        """Rebuild every chapter scene-by-scene from the existing spine. Used as an
        auto-escalation when scene_vs_summary / character_agency stay below floor."""
        spec, premise, bible = state.spec, state.premise, state.bible
        spine_by_num = {p.number: p for p in state.spine}
        rebuilt: list[Chapter] = []
        for plan in sorted(state.spine, key=lambda p: p.number):
            synopsis = "\n".join(f"Ch{c.number} ({c.title}): {c.summary}"
                                 for c in rebuilt if c.summary)
            prev_tail = " ".join(rebuilt[-1].prose.split()[-120:]) if rebuilt else ""
            avoid = self._avoid_list(RunState(idea=state.idea, chapters=list(rebuilt)))
            self._say(f"• Scene-level redraft chapter {plan.number}: {plan.title}")
            prose = self.draft_chapter_scenes(spec, premise, bible, plan, synopsis,
                                              prev_tail, avoid, spine_by_num.get(plan.number + 1))
            summary = self.summarize_chapter(plan, prose)
            rebuilt.append(Chapter(number=plan.number, title=plan.title, prose=prose,
                                   summary=summary))
            (self.run_dir / "chapters" / f"ch{plan.number:02d}.md").write_text(
                f"### Chapter {plan.number}: {plan.title}\n\n{prose}\n")
            self._save(state)
        state.chapters = rebuilt
        self._save(state)
        if self.use_derepeat:
            self.derepeat(state, rounds=2)
        manuscript = self.assemble(state)
        (self.run_dir / "manuscript.md").write_text(manuscript)
        self._say(f"✓ Scene-level redraft complete ({_wc(manuscript)} words)")
        return self.run_dir / "manuscript.md"

    # --- assembly ---
    def assemble(self, state: RunState) -> str:
        p, b = state.premise, state.bible
        out = [f"# {p.title}", "", "## Premise", "", p.paragraph, "", "## World / Characters", ""]
        for i, c in enumerate(b.characters, 1):
            desc = c.description or c.role
            out.append(f"{i}. {c.name}, {desc}")
        out += ["", "## Manuscript", ""]
        for ch in sorted(state.chapters, key=lambda c: c.number):
            out += [f"### Chapter {ch.number}: {ch.title}", "", ch.prose, ""]
        return "\n".join(out).strip() + "\n"

    def _avoid_list(self, state: RunState, k: int = 14) -> list[str]:
        """Phrasings to steer the next chapter away from — directly targets T1.4.

        Combines (a) 5-grams that have ALREADY repeated across chapters (the metric
        itself) and (b) the most distinctive long sentences from every drafted
        chapter, so the model stops parroting its own signature lines."""
        from collections import Counter

        from .evaluate.tier1 import _ngrams, _words

        if not state.chapters:
            return []
        per = [_ngrams(_words(c.prose)) for c in state.chapters]
        counts: Counter = Counter()
        for grams in per:
            for g in grams:
                counts[g] += 1
        repeated = [" ".join(g) for g, c in counts.most_common() if c >= 2][:k // 2]

        sents: list[str] = []
        for ch in state.chapters:
            for s in re.split(r"(?<=[.!?])\s+", ch.prose):
                if 9 <= _wc(s) <= 22:
                    sents.append(s.strip())
        sents.sort(key=_wc, reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for phrase in repeated + sents:
            key = phrase.lower()[:60]
            if key not in seen:
                seen.add(key)
                out.append(phrase)
            if len(out) >= k:
                break
        return out

    # --- orchestrator ---
    def run(self, idea: str, overrides: dict | None = None, resume: bool = False) -> Path:
        overrides = overrides or {}
        state_path = self.run_dir / "state.json"
        if resume and state_path.exists():
            state = RunState.from_dict(json.loads(state_path.read_text()))
            self._say(f"↻ Resuming run in {self.run_dir} ({len(state.chapters)} chapters done)")
        else:
            state = RunState(idea=idea, draft_model=self.model)

        if state.status == "init":
            self._say("• Inferring story spec…")
            state.spec = self.gen_spec(idea, overrides)
            state.status = "spec"
            self._save(state)
        spec = state.spec
        self._say(f"  spec: {spec.genre}/{spec.tone}, {spec.pov_label}, "
                  f"{spec.num_chapters}×~{spec.words_per_chapter}w, ending: {spec.ending}")

        if not state.premise:
            self._say("• Forging premise…")
            state.premise = self.gen_premise(idea, spec)
            self._save(state)
        self._say(f"  title: “{state.premise.title}” — {state.premise.logline}")

        if not state.bible:
            self._say("• Building world bible…")
            state.bible = self.gen_bible(idea, spec, state.premise)
            prot = next((c.name for c in state.bible.characters if c.role == "protagonist"),
                        state.bible.names()[0])
            spec.pov_character = prot
            self._save(state)
        self._say(f"  cast: {', '.join(state.bible.names())}")

        if not state.spine:
            self._say("• Outlining the spine…")
            state.spine = self.gen_spine(spec, state.premise, state.bible)
            self._save(state)
        self._say(f"  {len(state.spine)} chapters planned")

        done = state.drafted_numbers()
        spine_by_num = {p.number: p for p in state.spine}
        for plan in state.spine:
            if plan.number in done:
                continue
            self._say(f"• Drafting chapter {plan.number}/{len(state.spine)}: {plan.title}")
            prev_tail = ""
            if state.chapters:
                prev_tail = " ".join(state.chapters[-1].prose.split()[-120:])
            next_plan = spine_by_num.get(plan.number + 1)
            drafter = self.draft_chapter_scenes if self.scene_level else self.draft_chapter
            prose = drafter(spec, state.premise, state.bible, plan,
                            state.synopsis(), prev_tail, self._avoid_list(state),
                            next_plan)
            summary = self.summarize_chapter(plan, prose)
            ch = Chapter(number=plan.number, title=plan.title, prose=prose, summary=summary)
            state.chapters.append(ch)
            state.chapters.sort(key=lambda c: c.number)
            (self.run_dir / "chapters" / f"ch{plan.number:02d}.md").write_text(
                f"### Chapter {plan.number}: {plan.title}\n\n{prose}\n")
            self._save(state)
            self._say(f"    ✓ {_wc(prose)} words")

        state.status = "drafted"
        self._save(state)
        if self.use_derepeat:
            self.derepeat(state, rounds=3)
        manuscript = self.assemble(state)
        ms_path = self.run_dir / "manuscript.md"
        ms_path.write_text(manuscript)
        self._say(f"✓ Manuscript assembled: {ms_path} ({_wc(manuscript)} words)")
        return ms_path
