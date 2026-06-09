# Canon / continuity config — schema & intervention map

`story_writer` turns a story idea into a multi-chapter manuscript. A manual,
chapter-by-chapter continuity review of one generated manuscript surfaced a set
of recurring failure modes — names appearing before they should, the
protagonist's eye colour flipping, planning vocabulary leaking into prose,
timeline spans that don't add up, required artifacts going missing. This config
encodes the **rules that span chapters** as DATA so that *one* source of truth
both PREVENTS those failures at generation time and SCORES/GATES them at
evaluation time.

* **The schema is generic/per-project.** The pipeline writes other stories from
  other ideas; only the *values* are story-specific.
* **The instance** lives in `canon/<project>.yaml` (e.g.
  `canon/nameless_weapon.yaml`). YAML or JSON are both accepted.
* **The loader** is `canon.load_canon(path)` → `canon.Canon`.
* **Generation reads** `canon.render_chapter_slice(canon, chapter_number)`.
* **Evaluation reads** the `Canon` object directly:
  - deterministic metrics: `canon_metrics.py`
  - LM-judge metrics: `bench/judge.py`

## Schema

| Key | Shape | Consumed by |
|---|---|---|
| `project`, `title` | str | metadata |
| `narration.pov_rule` | str | slice (POV instruction) |
| `timeline.max_total_span_years` | int | `check_timeline_age` |
| `timeline.year_to_age` | {year_code:int → age:int} | slice / reference |
| `timeline.year_codes` | {code → {year, note}} | reference |
| `timeline.key_spans` | list of {from_code, to_code, years, note} | reference |
| `name_arc.earliest_narration` | {name → earliest chapter int} | slice + `check_name_arc` |
| `name_arc.learned` | {name → chapter int} | reference |
| `name_arc.reveal_rules` | list of {speaker, name, from_chapter} | `check_name_arc` |
| `name_arc.call_order` | {chapter int → [name, …]} | reference / judge |
| `locked_appearance.protagonist` | {eyes, hair, hair_style, build, permanent_marks, principle, contradictions} | slice + `check_canon_fact` + judge |
| `required_artifacts` | {chapter int → {name, patterns:[…]}} | slice + `check_required_artifact` |
| `canon_invariants` | list of {id, rule, after_chapter?} | slice + judge |
| `accepted_aliases` | {spoken form → canonical} | name-drift metrics |
| `motif_allowlist` | list of sanctioned refrains | phrase-reuse metrics |
| `scaffolding_blocklist` | list of regex (case-insensitive) | slice + `check_scaffolding_leak` |
| `device_allowlist` | list of sanctioned meta-narrative phrases | `check_scaffolding_leak` |

`locked_appearance.protagonist.contradictions` maps each locked feature to the
adjectives that would contradict it (e.g. `eyes: [blue, green, …]`), which is
what the canon-fact metric scans for.

## Intervention map (finding → PREVENT → GUARD)

Every finding is wired both ways: a generation-time injection that prevents it,
and a metric that guards it in eval.

| Finding (from the review) | PREVENT — injected into | GUARD — metric | Kind |
|---|---|---|---|
| Scaffolding vocab in prose ("Ch.19", "Beat 3", "POV", "the reader") | per-chapter draft slice + critic→revise (`canon_guard`) | `check_scaffolding_leak` | deterministic |
| A name used before its allowed chapter (Elias ≥ Ch.7, Severin ≥ Ch.11; Marta says "Elias" only from Ch.17) | slice "names not yet revealed" + revise | `check_name_arc` | deterministic |
| Appearance contradicts locked anchor ("blue eyes" vs grey) | slice locked-appearance + revise | `check_canon_fact` | deterministic |
| Required artifact missing (hymn Ch.4, bard song Ch.12) | slice "required this chapter" + revise | `check_required_artifact` | deterministic |
| Timeline/age span impossible ("the next eight years", span 5) | slice timeline facts | `check_timeline_age` | deterministic |
| Protagonist barely physically described | slice "ground the POV character physically" step | `judge.appearance_described` + `appearance_vs_spec` | LM |
| Azazel on the surface after Ch.11 | slice canon invariant | `judge.azazel_maw` | LM |
| Command-of-the-Void leash not fraying through Acts IV–V | slice canon invariant | `judge.leash_fray` | LM |
| Kira reduced to a strawman | slice canon invariant | `judge.kira_not_strawman` | LM |
| POV discipline | slice POV rule | `pov_check` / `judge.pov` | LM |
| Dangling setup/payoff | — | `judge.dangling_payoff` | LM |

`accepted_aliases`, `motif_allowlist`, and `device_allowlist` are *anti–false-positive*
data: the metrics read them so that a spelled-out "Vessel Eighty-Four", a
sanctioned refrain, or the device "The reader sees her stand." is not mistaken
for drift / reuse / a scaffolding leak.

## Layers

* **Layer A — PREVENT (generation):** `canon.render_chapter_slice` is injected
  into the per-chapter drafting signatures; `canon_guard` rejects drafts that
  violate the cheap deterministic constraints and revises with the violation as
  feedback.
* **Layer B — OPTIMIZE (eval/optimizer):** `canon_metrics` + `bench/judge` are
  composed into a weighted metric (`bench/metric.py`) used by `dspy.Evaluate`
  and an optimizer over a regression set seeded with the real bugs as gold
  negatives.
* **Layer C — CATCH (post-gen gate):** the same `canon_metrics` code runs in
  `bench/criteria.score` and reports through `qa.py`'s info/warn/fail buckets
  with thresholds.
