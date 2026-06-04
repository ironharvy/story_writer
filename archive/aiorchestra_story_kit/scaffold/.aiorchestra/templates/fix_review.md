The editorial review flagged issues with your change for issue #{number}.

Issue #{number}: {title}

{body}

Review feedback:

{errors}

Address each point with the smallest correct change, keeping everything consistent with the rest of `story/`:

- continuity / canon contradiction → fix the changed artifact. Never rewrite the canon files (`story/idea.md`, `story/premise.md`, `story/spine.md`, `story/world/*`, earlier chapters) to match a mistake; if the canon itself is the problem, that is a separate `Revise — …` issue, not this one.
- "doesn't deliver the phase" → produce the artifact the issue title asks for.
- style / pacing notes → revise the prose; follow the `## Style` section in `AGENTS.md`.
- process hygiene (STATUS / notes not updated, wrong labels on the next issue, meta commentary in story files) → fix it.
- never weaken `story/_check.py` or `.aiorchestra/` config to resolve feedback.

Do NOT run tests — just fix it.
