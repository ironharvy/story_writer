---
name: "Phase 11 — Assemble & Publish"
about: "Compile story/final.md and publish it"
title: "Assemble & Publish: <story title>"
labels: ["aiorchestra", "claude", "story"]
---
**Phase 11, terminal (see `WORKFLOW.md`).** Build `story/final.md` with sections in order: a `# <story title>` heading, `## Core Premise`, `## Spine`, `## World Bible` (Rules / Characters / Locations / Timeline), `## Chapter Plan`, `## Enhancers Guide`, `## Final Story` (every chapter with its title and full prose, in order).

If `scripts/run_qa.py` exists in the repo, run it on `story/final.md` and fix clear issues (name drift, character presence, cross-chapter phrase reuse).

Then publish: follow `.claude/skills/herenow/SKILL.md` to publish `story/final.md`; record the returned `site_url` and `claim_url` in `story/STATUS.md` and the PR body. If `.claude/skills/herenow/` is not present, skip publishing and say so in `story/STATUS.md`. Anonymous publishes are claimable for 24h; set `HERENOW_API_KEY` in the run's environment to publish under an account.

Do **not** file a follow-up issue — this is the end of the chain.

Read first: everything in `story/`.
