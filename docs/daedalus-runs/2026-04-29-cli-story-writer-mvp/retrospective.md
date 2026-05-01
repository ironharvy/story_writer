# Retrospective

## Outcome

Partial / not customer-acceptable: the CLI MVP structure, contracts, tests, local artifact operations, and a live Ollama generation path were exercised, but the generated story does not meet the product acceptance bar. A real customer would not accept the output because QA found a hard continuity defect and the prose repeats chapter openings in a visibly mechanical way.

## Causes

- Architecture drift existed between README and new living docs before the independent review.
- Existing sample artifacts came from an older schema and included broken prose.
- Initial QA proper-noun detection was too naive for strict mode.
- The sandbox initially blocked localhost access; escalated command execution confirmed Ollama was reachable.
- The first live run exposed a parser tolerance bug for title-cased enum values.
- The live story exposed real QA findings: invented continuity (`Liora`) and repeated chapter openings.
- The process initially over-weighted executable infrastructure and under-weighted the product's actual success metric: coherent, interesting prose.
- The phrase "complete with known issues" was too weak for a creative-output MVP where the known issues invalidate the primary value proposition.

## Follow-Up

- Improve deterministic unknown-name fallback prose so the replacement reads naturally.
- Prevent upstream chapter plans from asking prose to reveal a specific name when the world bible has not named the entity.
- Replace or delete the broken legacy demo run after a successful clean generation.
- Resolve SPIKE-001 with measured local model quality.
- Treat future story-generation runs as failed or partial until a rendered story passes structural validation, QA, and human artifact inspection against the prose-quality bar.
