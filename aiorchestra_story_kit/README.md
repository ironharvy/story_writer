# aiorchestra story kit

Turn an (almost) empty git repo into an **aiorchestra-driven story project**:
every writing phase is a GitHub issue, the agent opens a PR carrying exactly one
artifact, you review/merge it, then it files the next phase's issue. The phase
chain mirrors the `story_writer` `write-story` skill — idea → interrogatives →
premise → spine → world bible → chapter plan → enhancers → chapters →
assemble & publish.

Full algorithm: **[scaffold/WORKFLOW.md](scaffold/WORKFLOW.md)** — it gets
copied into every story repo so the agent can read it there.

## What's in here

| Path | Purpose |
|---|---|
| `init_story_repo.py` | One-shot scaffolder. stdlib only; needs the `gh` CLI. |
| `scaffold/` | Files copied verbatim into the target repo (`WORKFLOW.md`, `.aiorchestra/`, `.github/ISSUE_TEMPLATE/`, `story/`). |
| `repo_md_templates/` | `AGENTS.md` / `CLAUDE.md` templates, rendered from the target repo's README. |

## Prerequisites

- `gh` CLI, authenticated (`gh auth login`). The target repo can already exist
  on GitHub, or the init script can create it for you. Not needed if you run
  with `--no-issue`.
- An [aiorchestra](https://github.com/ironharvy/AIOrchestra) instance running
  somewhere with visibility into the target repo (e.g. `aiorchestra dispatch
  --watch` on your home machine), `claude-code` provider configured.
- Python 3.10+.
- Optional: a `herenow-skill/` directory available next to this kit (it ships in
  `story_writer`) — the init script vendors it into the new repo's
  `.claude/skills/herenow/` so the "Assemble & Publish" phase can publish.

## Quickstart

```bash
mkdir my-story && cd my-story
python /path/to/aiorchestra_story_kit/init_story_repo.py . --push
```

The script is interactive and walks you through the missing pieces: if the
directory isn't a git repo yet it offers to `git init`; if there's no
`README.md` it offers to create a starter one (asking for a title and logline);
if the repo isn't on GitHub yet it offers to run `gh repo create`. Already have
a README written out? Even better — it parses the `# Title` line and the first
paragraph as the logline. Pass `--yes` (or `-y`) to accept every prompt
non-interactively, or `--no-issue` to just scaffold + commit without touching
GitHub.

A fully manual setup still works if you prefer it:

```bash
mkdir my-story && cd my-story && git init

cat > README.md <<'EOF'
# The Last Keeper

A lighthouse keeper on a dying coast discovers the light is the only thing
holding back something in the fog; the mainland wants the lighthouse
decommissioned.
EOF
git add -A && git commit -m "seed idea"
gh repo create my-story --private --source=. --remote=origin --push

python /path/to/aiorchestra_story_kit/init_story_repo.py . --push
```

The script copies `scaffold/` in, renders `AGENTS.md` / `CLAUDE.md` from your
README, seeds `story/idea.md` with the README text, vendors the herenow skill
(if found), commits, ensures the `aiorchestra` / `claude` / `story` /
`next-phase` labels exist, pushes, and opens issue **#1 — "Idea: The Last
Keeper"** labelled `aiorchestra` + `claude` + `story`.

From there: aiorchestra discovers the `aiorchestra`-labelled issue, the agent
opens a PR with `story/idea.md` and files the next issue (`Premise: …`,
labelled `story` + `next-phase` only). You review/merge the PR, add
`aiorchestra` + `claude` to that next issue, and repeat down the chain.

## Notes & caveats

- aiorchestra's `dispatch` mode discovers issues by the **`aiorchestra`** label
  (it isn't configurable); the agent family is resolved from the `claude` label
  (or the configured default). The kit's `.aiorchestra/config.yaml` only
  overrides test / review / CI / OSINT settings — not the label.
- If you ever scaffold *into* a repo that already runs a GitHub Action on the
  `claude` label (e.g. `story_writer`'s `agent-implement.yaml`), disable that
  workflow first, or both will fire on the same issue.
- The agent needs `gh` (or an equivalent) **inside the aiorchestra run** to file
  the next phase's issue. If it can't, it writes the next-issue spec into
  `story/STATUS.md` under a `## Next step` heading — file it yourself.
