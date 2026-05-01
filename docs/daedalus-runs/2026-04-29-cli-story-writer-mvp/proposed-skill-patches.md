# Proposed Skill Patches

- During Phase 6, require reviewers to test existing sample artifacts as well as newly produced code paths.
- In local-first projects, explicitly validate that documented local services are reachable before treating integration tests as skipped-but-acceptable.
- Add a standard checklist item for destructive/stale artifact risks when a CLI creates named local run directories.
- Tighten Daedalus outcome labels: if the primary customer-facing artifact fails the stated value proposition, classify the run as `partial` or `failed`, not `complete with known issues`, even when code, tests, and infrastructure work.
- For creative/content products, require the output grader to answer one explicit customer-acceptance question: "Would the target user accept this artifact as satisfying the MVP promise?" If the answer is no, the delivery gate fails.
- In Review & Delivery, separate "backend path verified" from "product accepted." A real LLM round trip proves integration only; it does not prove the generated artifact is good enough.
- When QA detects a hard defect in generated content, Daedalus should either loop back into a correction sprint or mark the run partial. Detection-only is a valid engineering milestone, but not a shippable product outcome when the defect violates the product goal.
- Add a postmortem prompt that asks whether the orchestrator over-valued structural checks because they were easier to validate than the actual customer value.
