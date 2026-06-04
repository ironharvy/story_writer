Your previous change for issue #{number} failed the story consistency check.

Issue #{number}: {title}

{body}

Check output:

{errors}

Fix it with the smallest correct change:

- "prerequisite X is empty/placeholder" → the artifact you wrote depends on a `story/` file that has no real content. Do NOT fake the prerequisite and do NOT weaken `story/_check.py`. If the prerequisite genuinely has not been written yet, this phase should not run yet — output `NEEDS_CLARIFICATION: <which prerequisite phase still needs to be done>` and make no changes.
- "story/final.md missing the `## Final Story` section" / "too short" → assemble it properly per the `Assemble & Publish` instructions in `.aiorchestra/templates/implement.md`.
- "story/chapters/<file> is suspiciously short" → that chapter is a stub; draft it as a full chapter.
- anything else → address exactly what the message says, touching only what is needed.

Do NOT run tests — just fix it. Keep `story/STATUS.md` accurate.
