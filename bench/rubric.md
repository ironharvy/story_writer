# Tier-3 Qualitative Rubric (LLM judge, 1–5)

Implemented by the Tier-3 judge in `bench/judge.py` (`QualitativeRubric`).
Threshold to ship: **average ≥ 4.0 with no axis below 3.**

| Axis | 1 | 5 |
|---|---|---|
| **Arc & structure** | episodic, no shape | clear setup → rising action → climax → resolution |
| **Ending** | stops / fizzles / deus ex machina | earned, lands the premise's promise |
| **Character agency** | events happen *to* passive figures | choices drive the plot, characters want things |
| **Scene vs summary** | tells/summarizes throughout | dramatizes key beats in scene |
| **Prose quality** | flat, repetitive, garbled | varied, controlled, few tics |
| **Cohesion** | threads dropped, setups unpaid | setups pay off, threads resolve |

## Judge instructions

- Use a local Ollama model first; only escalate to the hosted fallback when
  verdicts are unstable/garbled.
- Prefer an independent judge — different from the model that wrote the
  draft — to avoid self-preference bias.
- Feed the judge the *narration only*; quoted dialogue shouldn't confound
  the prose-quality / scene-vs-summary axes.
- Return `{axis: {score: int 1–5, note: str}}` per axis plus an overall note.

See `bench/eval-spec.md` for the full evaluation spec including Tier-1
(deterministic, implemented in `bench/criteria.py`) and Tier-2 (POV /
protagonist naming / contradictions / premise fidelity — also LLM-judged).
