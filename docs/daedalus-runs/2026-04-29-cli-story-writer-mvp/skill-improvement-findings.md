# Skill Improvement Findings

## Core Finding

This run exposed a Daedalus process weakness: the process can drift into treating a technically executable system as a successful delivery even when the customer-facing artifact fails the MVP promise.

For this project, the MVP promise is high-quality story prose. The live Ollama run proved the pipeline can execute, persist artifacts, render Markdown, and run QA. It did not prove the product is acceptable. The generated story contained a hard continuity defect (`Liora` appears outside the world bible) and repeated chapter openings that make the prose feel mechanical.

## Skill Lesson

Daedalus needs a stronger distinction between:

- **Engineering path verified**: commands run, tests pass, dependencies work, artifacts are created.
- **Product value delivered**: the primary artifact satisfies the user's success criteria.

If the primary artifact fails, the run must be classified as `partial` or `failed` from a delivery standpoint. "Complete with known issues" is only appropriate when the known issues do not invalidate the core value proposition.

## Proposed Skill Changes

1. Add a mandatory Phase 6 customer-acceptance verdict:
   - "Would the target user accept this output as meeting the MVP promise?"
   - Allowed answers: `yes`, `no`, or `not inspectable`.
   - If `no`, the run cannot be marked complete.

2. Add a specific creative-output gate:
   - Read the rendered artifact end-to-end.
   - Identify continuity breaks, duplicated passages, empty sections, truncated sections, invented facts, and prose that feels visibly mechanical.
   - If any issue violates the product's primary promise, loop back or mark partial.

3. Require outcome labels to be value-based:
   - `success`: product artifact meets the acceptance bar.
   - `partial`: engineering path works, but artifact quality or feature completeness misses the bar.
   - `failed`: primary path cannot run or artifact is unusable.

4. Local-service validation should handle sandbox boundaries:
   - If a local dependency fails with permission/socket symptoms, retry through the approved escalation path before calling it unavailable.

5. Detection-only QA should not be mistaken for delivery:
   - If hard QA findings remain, either run a correction sprint or explicitly classify as partial/failed.

6. Make PM and Tech Lead awareness operational:
   - PM owns the release classification and backlog pivot when customer acceptance fails.
   - Tech Lead owns root-cause analysis and the revised technical strategy.
   - Awareness is not a chat note; it must be represented by a decision entry, a changed backlog, and a release-blocking status.
