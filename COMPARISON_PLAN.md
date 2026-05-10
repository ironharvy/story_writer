# Comparison plan: the 3 story-drafting variants

Handoff brief for an agent running locally against Ollama. Goal: run all three
chapter-drafting strategies on identical inputs, score them, and recommend
which to keep / merge / drop.

Branch: `claude/world-state-implementation-UN9s6` — `git pull origin claude/world-state-implementation-UN9s6` before starting.

## The variants

All three are identical up to **and including** world-bible generation
(`build_world_bible`); they differ only in how chapters are drafted afterward.

| | Entry point | Drafting code | Continuity between chapters |
|---|---|---|---|
| **A · baseline** | `python mymain.py` | `pipeline.py` | per-chapter `run_enhance_chapter` + rolling text summary (`run_generate_story_so_far`) |
| **B · world-state** | `python mymain_ws.py` | `pipeline_ws.py` + `world_state.py` | structured `WorldState` (story clock / characters / locations / plot threads / key objects / recent events), updated after each chapter |
| **C · dspy.Module** | `python mymain_module.py` | `pipeline_module.py` → `story_module.WriteStory` | *none* — each chapter drafted from its own outline beats + world bible + act-sliced spine (DSPy `DraftArticle` style) |

## Step 0 — make runs unattended

All three prompt via `ui.review_answer` (Rich `Confirm.ask(..., default=True)`)
for every foundation step (idea, premise, spine, each rule/location/character…),
and A & B also for the chapter plan and every chapter — ~30-40 prompts per run.
No code change needed: pipe an endless stream of blank lines, so every confirm
takes its default (accept the proposed answer). Accepted answers never trigger
the follow-up `Prompt.ask`, so blank input is sufficient.

- Use `yes '' | …` (blank lines), **not** `< /dev/null` — EOF on stdin can make
  Rich misbehave; an endless stream of blank lines is safe.
- This means every proposed answer is accepted and every confirm takes its
  default — exactly what you want for a clean A/B (no human-introduced
  divergence). Don't pipe if you ever want to hand-edit something mid-run.
- `yes` getting SIGPIPE when Python exits is normal/harmless; the pipeline's
  exit code is Python's. Piping stdin also makes Rich non-interactive (no
  color) — fine.

## Step 1 — fixed inputs

Same `--idea` / `--title` / `--number-of-chapters` for all three. Suggested:

```
--idea "A lighthouse keeper on a dying coast discovers the light is the only thing holding back something in the fog; the mainland wants the lighthouse decommissioned." --title "The Last Keeper" --number-of-chapters 7
```

## Step 2 — warm the shared cache (recommended)

Keep the DSPy disk cache **on** (default). Run **A first**: the foundation calls
(clarify / premise / spine / world bible) and the chapters-plan call use
identical signatures + inputs in A and B, so B reuses them from cache →
byte-identical foundation, isolating the drafting difference. C diverges at the
outline (new `StoryOutline` signature) but still shares idea / premise / spine /
world-bible from cache.

## Step 3 — run all three

The `--output-file` default is shared across the three entry points, so you
**must** override it each time.

```bash
mkdir -p .tmp/cmp
yes '' | python mymain.py        --idea "<IDEA>" --title "<TITLE>" --number-of-chapters 7 --num-ctx 24576 --output-file .tmp/cmp/A_baseline.md   -v
yes '' | python mymain_ws.py     --idea "<IDEA>" --title "<TITLE>" --number-of-chapters 7 --num-ctx 24576 --output-file .tmp/cmp/B_worldstate.md -v
yes '' | python mymain_module.py --idea "<IDEA>" --title "<TITLE>" --number-of-chapters 7 --num-ctx 24576 --output-file .tmp/cmp/C_module.md     -v
```

Use whatever Ollama model you're running (`--model qwen3 --provider ollama`,
etc.). Logs land in `.tmp/mymain.log`. If the log shows context truncation,
bump `--num-ctx`; if a run OOMs, lower `--num-ctx` / `--max-tokens` or use
`--number-of-chapters 5` — and **re-run all three** with the same settings so
they stay comparable.

## Step 4 — automated metrics

```bash
python scripts/run_qa.py .tmp/cmp/A_baseline.md .tmp/cmp/B_worldstate.md .tmp/cmp/C_module.md
python scripts/word_count.py .tmp/cmp/A_baseline.md      # repeat for B and C
```

Per variant, record:
- `# FAIL` / `# WARN` findings from `run_qa.py` — especially `check_name_drift`,
  `check_character_presence`, `check_cross_chapter_phrase_reuse`.
- Top repeated content words (`word_count.py`).
- Total word count + per-chapter word counts (consistency).
- Wall-clock time, and total tokens if a token callback / Langfuse is active.

## Step 5 — read-through rubric (the real test)

Read all three "Final Story" sections. Score each 1-5 on:

1. **Continuity** — characters / places / objects stay consistent
   chapter-to-chapter; setups pay off.
2. **Coherence** — one story vs. 7 disconnected episodes.
3. **Structure** — follows the spine / 3-act arc; the ending lands.
4. **Prose quality** — varied rhythm, concrete scenes, no purple boilerplate or
   reused phrasing.
5. **Plot-thread tracking** — early open questions get resolved.

Note 2-3 concrete good/bad examples per variant.

## Step 6 — write up

`.tmp/cmp/REPORT.md`: metrics table + rubric scores + standout examples + a
recommendation (which strategy to keep / merge / drop, plus quick wins — e.g.
"C drifts on names → also feed it the full outline into `DraftChapter`").

## Notes

- `.tmp/` is gitignored — outputs won't be committed (fine). If you want the
  report in the repo, put it elsewhere and say so.
- The variants are isolated files; nothing in `story.py` / `pipeline.py` needs
  touching.
