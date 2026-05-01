# PM / Tech Lead Investigation: Prose Quality Release Blocker

## Status

**Release blocked / partial.** The CLI pipeline executes against local Ollama and produces persisted artifacts, but the rendered story from `runs/smoke-ollama-real` is not customer-acceptable for the MVP promise of coherent, interesting, artifact-free prose.

This is not an infrastructure failure. It is a product-quality failure in the primary customer artifact.

## Evidence Reviewed

- `runs/smoke-ollama-real/world_bible.json`
- `runs/smoke-ollama-real/chapter_plan.json`
- `runs/smoke-ollama-real/enhancement.json`
- `runs/smoke-ollama-real/chapters/*.json`
- `runs/smoke-ollama-real/qa/*.json`
- Rendered story at `/tmp/smoke-ollama-real.md`

## QA Findings

**Hard finding: invented named entity**

- `runs/smoke-ollama-real/qa/03.json` flags `Liora` under R5.
- `Liora` does not appear in the world bible, chapter plan, or enhancement guide.
- The world bible only defines `Elara Voss`, `The Shifting City`, `The Archivist`, `The Forgotten Archive`, and `The Shifting City`.
- Chapter 3 prose invents: `*Liora.* Her mother's name.`

**Soft finding: repeated prose**

- `runs/smoke-ollama-real/qa/04.json` and `qa/05.json` flag cross-chapter repeated sentences.
- Chapters 4 and 5 reuse core sentences from chapter 3, including:
  - `She gritted her teeth, forcing her grip tighter, but the ink bled into the margins...`
  - `The air in the archive grew heavier...`
  - `The city's labyrinthine streets, half-formed in her mind...`

**Human read-through finding**

- Chapters 3-5 feel mechanically similar. The repeated openings make the story read like regenerated variants of the same scene rather than distinct progression beats.

## Root-Cause Assessment

### Invented `Liora`

Likely cause: prose-stage model invention.

Reasoning:

- The world bible says Elara's family disappearance is tied to the city, but does not name the missing family member.
- The chapter plan says only `her family member's name`.
- The enhancement guide says only `the family member's name in the mist`.
- The prose model filled that unspecified slot with `Liora`.

This is a world-bible grounding gap: downstream prose had permission to dramatize an unnamed fact, but no contract telling it to keep the fact unnamed or request/update the world bible before naming it.

### Repeated Chapter Openings

Likely cause: weak chapter-specific prose contract plus overlapping inputs.

Reasoning:

- Chapters 3, 4, and 5 all concern mapping, dissolution, legacy, and surrender.
- The chapter plan and enhancement guide repeat similar motifs: archive, map, city dissolution, Archivist warning, legacy as echoes.
- `ProseSignature` only asks for polished prose and supplies `prior_summary`, but it does not explicitly forbid reusing prior openings, images, sentence structures, or scene beats.
- QA detects repeated full sentences after generation, but no revision loop acts on those findings.

### Customer Acceptance Failure

Likely cause: the process validated execution before validating reader value.

Reasoning:

- The CLI path, persistence, rendering, and QA all work.
- The output still fails the MVP promise because the artifact is not coherent enough for a real reader.
- This confirms the PM/TL pivot in DEC-0020: infrastructure completion is not the product.

## Role Conclusions

### PM

Keep release blocked. The next sprint must target prose quality and acceptance gates only. Do not add web service work, UI, sharing, telemetry, or new product surface until a live rendered story passes QA and human read-through.

PM acceptance criteria for the next sprint:

- No hard QA findings in the rendered story.
- No repeated chapter openings across adjacent or later chapters.
- No named entity appears in prose unless it exists in the world bible or is intentionally added through a reviewed bible update.
- Each chapter must clearly advance its own chapter-plan purpose.
- A human read-through must judge the story customer-acceptable for a first MVP draft.

### Tech Lead

The next implementation should focus on three technical controls:

1. **Prose grounding contract**
   - Prose must not invent names for unnamed world-bible entities.
   - If the plan says `family member` without a name, prose must preserve the generic reference or fail/revise.

2. **Anti-repetition context**
   - Pass prior chapter openings or first paragraphs into the prose stage.
   - Explicitly forbid reusing prior opening situation, sentence structure, and signature imagery.
   - Add a QA rule that flags repeated openings directly, not only repeated full sentences.

3. **Revision loop for hard QA**
   - Hard QA findings should trigger a controlled rewrite of the affected chapter or block release.
   - The rewrite must use the world bible and QA spans as constraints.
   - It must not silently update the world bible.

### QA

QA should keep R5 hard. The `Liora` finding is legitimate and release-blocking.

QA should add a separate repeated-opening rule because the current R8 only catches exact sentence reuse. The human-visible defect can exist even when sentences are paraphrased.

### BA

The acceptance language should be sharpened from "interesting to read" into operational criteria:

- The prose must not introduce new named entities outside the world bible.
- Chapters must not read like variants of the same scene.
- Chapter openings must be distinct in setting action, emotional posture, or narrative motion.
- Every chapter must deliver its stated chapter-plan purpose.

## Recommended Next Sprint

1. Add repeated-opening detection.
2. Strengthen `ProseSignature` with:
   - no invented named entities,
   - world-bible-only facts,
   - distinct opening requirement,
   - prior-opening avoidance input.
3. Add a hard-QA revision loop for unknown proper nouns.
4. Run a model-routing comparison using the same artifacts:
   - `qwen3:latest` for prose,
   - `gemma4:26b` for prose.
5. Run a fresh live smoke and classify outcome by customer acceptance, not command completion.

## Customer-Acceptance Verdict

**No.** A target user would not accept the current smoke story as satisfying the MVP promise. The system is technically executable, but the story output is not yet a shippable product artifact.
