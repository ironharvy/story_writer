# Validation Summary

## Delivery Classification

Partial / not customer-acceptable. The system can execute a real local-Ollama run and produce artifacts, but the generated story fails the MVP's core product criterion: coherent, artifact-free prose a reader would accept.

## Passed

- `ollama list` with escalated localhost access -> `gemma4:26b`, `qwen3:latest`, `deepseek-r1:7b`, `gemma3:4b`, `gemma3:latest`
- `curl -sS --max-time 3 http://localhost:11434/api/tags` with escalated localhost access
- `.venv/bin/ruff check .`
- `.venv/bin/ruff format --check .`
- `.venv/bin/python -m pytest -q` -> 10 passed, 1 skipped
- `.venv/bin/story-writer --help`
- `.venv/bin/story-writer render demo-hollow --out /tmp/story-writer-demo-hollow.md`
- `.venv/bin/story-writer new --idea test --profile invalid --non-interactive` cleanly rejects invalid profile
- `.venv/bin/story-writer new --idea "A cartographer finds a city that appears only in unfinished maps." --slug smoke-ollama-real --length short --non-interactive --model qwen3:latest --embellish-probability 0` completed against local Ollama after enum normalization.
- `.venv/bin/story-writer render smoke-ollama-real --out /tmp/smoke-ollama-real.md` produced a readable Markdown story.
- `.venv/bin/story-writer new --idea "A cartographer finds a city that appears only in unfinished maps." --slug smoke-quality-fix-03 --length short --non-interactive --model qwen3:latest --embellish-probability 0` completed against local Ollama after the prose-quality fix sprint.
- `runs/smoke-quality-fix-03/qa/*.json` has no failed QA rules.
- `/tmp/smoke-quality-fix-03.md` is readable, non-empty, and no longer has the repeated chapter-opening defect seen in `smoke-ollama-real`.

## Not Passed

- Historical smoke `smoke-ollama-real` is not clean. `runs/smoke-ollama-real/qa/03.json` flags a hard continuity issue: `Liora` appears in prose but is not in the world bible. Chapters 4 and 5 also have soft cross-chapter repeated sentence findings.
- Human artifact inspection found the repeated openings in chapters 3-5 visibly mechanical. The output is readable, but not shippable as a customer-facing story draft.
- Current smoke `smoke-quality-fix-03` passes QA, but human inspection found one awkward fallback phrase in chapter 3: `a name she could not read`. This is not release-blocking at the QA-rule level, but it should be improved by addressing the upstream chapter-plan wording that asks prose to reveal a family member's name without a world-bible name.

## Known Artifact Issue

- `runs/demo-hollow` is a legacy sample with visibly broken prose: Chapter 3 is blank and Chapter 4 is truncated. It is renderable after compatibility fixes, but it is not evidence that the current live generation path produces coherent prose.
