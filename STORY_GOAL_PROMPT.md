# Story Generator — `/goal` prompt

The goal brief to hand a local Claude Code instance (running against Ollama).
Drop `STORY_EVAL_SPEC.md` into the target repo's root first; this prompt points
at it. Requires network access (for the craft-research step) and a reachable
Ollama at `http://localhost:11434`.

---

```
/goal Build a program that turns a user's story idea into a cohesive, interesting, complete manuscript.

WHAT TO BUILD
Design and implement the algorithm/workflow yourself — don't just call a model once.

First, research the craft: use the internet to gather best practices on how the story-writing and manuscript-development process actually works (e.g. premise/logline, three-act and other structures, the snowflake method, beat sheets, character arcs, scene-vs-sequel, outlining vs discovery writing, revision passes). Summarize what you learn and let it inform the design.

Then think out of the box: propose several competing algorithm designs — not one obvious pipeline. Compare them on tradeoffs (coherence, surprise, cost/speed on a local model, robustness), pick the most promising, and say why. Keep the runners-up noted so you can fall back to or hybridize them if the chosen one underperforms on the eval bar. You own the design and may change it as evidence comes in.

A reasonable baseline shape if you need a starting point: clarify the idea → premise → outline/spine → world bible (cast + locations + rules) → per-chapter plan → draft chapters → self-evaluate → revise. Beat it if you can.

The program is interactive: when the user's idea is underspecified (genre, length, tone, POV, ending), it asks a few clarifying questions before generating, then proceeds. Write outputs incrementally to disk so an interrupted run keeps everything produced so far, and support resuming a run.

MODEL STRATEGY (important)
The generation pipeline must use a local Ollama model as its LLM backend. Iterate on the fast model `qwen3:latest` while building the workflow (≈5 min end-to-end, prose quality doesn't matter for "does it run"), then validate real output on the quality model `qwen3.6:27b`. Only escalate to the hosted paid model `deepseek/deepseek-v4-pro` (via litellm, `DEEPSEEK_API_KEY`) when the local model repeatedly cannot clear the evaluation bar — and keep that usage minimal because it costs money. Always note in your scorecard which model produced a result. Ollama is at http://localhost:11434. Pin a generous max-tokens (8192+) so verbose models don't truncate and crash.

EVALUATION (your objective function)
Read STORY_EVAL_SPEC.md and implement it as a runnable eval harness that emits a JSON scorecard (each check → severity + message, rubric scores, and a final ship:true|false). Tier-1 checks are deterministic (no LLM); Tier-2/Tier-3 use the local Ollama model as judge (escalate the judge only if its verdicts are unstable). A manuscript "ships" only when it clears every hard gate and the rubric threshold defined there. Re-run the harness after every meaningful change — measure, don't guess. Build the regression fixtures listed at the bottom of the spec so fixed defects stay fixed.

CHECKPOINTS & EXPERIMENTATION
Use git as your checkpoint system. Commit each working state (pipeline that runs, first draft that clears Tier-1, first draft that ships). Before trying a risky redesign — a different drafting strategy, a new continuity approach, or one of your runner-up algorithm designs — branch or tag the last good checkpoint so you can revert and compare instead of losing it. Never discard a checkpoint that scores better than the current attempt. Treat the eval scorecard as the fitness signal for which checkpoint wins.

DELIVERABLES
1. A runnable program (clear CLI/entry point + README) that generates a manuscript from an idea.
2. The eval harness implementing STORY_EVAL_SPEC.md.
3. At least one sample manuscript that clears the ship gate, with its scorecard committed.
4. A short design note: the algorithm options you considered, what you learned from researching the craft, and why you chose the one you did.

Ask me any clarifying questions about this goal before you start.
```
