#!/usr/bin/env bash
#
# bootstrap_story_writer.sh
# -------------------------
# Scaffolds a fresh, intentionally-lean "Story Writer" experiment repo:
# a from-scratch web service that uses an LLM to write stories on request,
# with free and paid tiers. The seed is deliberately minimal so we can
# observe what Claude Code builds on its own (architecture, evaluation,
# testing, model choice) when given the product goal and a shelf of
# optional tools.
#
# Usage:
#   ./bootstrap_story_writer.sh [target-dir]
#
# Env toggles (optional):
#   CREATE_GITHUB_REPO=1   # if `gh` is installed, create + push a private repo
#   GH_VISIBILITY=private  # private (default) | public
#
# What it creates:
#   README.md            - one-paragraph product brief (no architecture)
#   CLAUDE.md            - the "tool shelf": optional tools + collaboration norms
#   EXPERIMENT_LOG.md    - decision journal (seeded with a single entry)
#   .env                 - real env file with EMPTY placeholders (git-ignored)
#   .env.example         - committed template of the same keys
#   .gitignore
# Then: git init + initial commit, and prints the /goal + ultracode kickoff.

set -euo pipefail

TARGET_DIR="${1:-story-writer}"
TODAY="$(date +%F)"

if [[ -e "$TARGET_DIR" ]]; then
  echo "Refusing to overwrite existing path: $TARGET_DIR" >&2
  exit 1
fi

echo "==> Creating $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# ---------------------------------------------------------------------------
# README.md  — product brief only. No tech stack, no architecture: that is the
# whole point. Whatever Claude decides should be discoverable from the build.
# ---------------------------------------------------------------------------
cat > README.md <<'EOF'
# Story Writer

A web-based service that turns a user's request into a finished story, written
by an LLM.

Users describe what they want — a premise, a genre, a vibe, a length — and the
service writes them a story. Two tiers:

- **Free** — limited number of stories, shorter length, standard quality.
- **Paid** — more stories, longer / multi-chapter works, higher-quality
  generation, and extras.

This repository is an open-ended build. The goal is the product above. *How* to
get there — architecture, framework, model choice, evaluation, and testing — is
yours to decide. See **CLAUDE.md** for what's available in the environment, and
record notable decisions in **EXPERIMENT_LOG.md**.
EOF

# ---------------------------------------------------------------------------
# CLAUDE.md — the ambient capability shelf. Everything here is OPTIONAL and
# phrased as "available, use your judgment". Nothing prescribes architecture.
# ---------------------------------------------------------------------------
cat > CLAUDE.md <<'EOF'
# Working in this repo

This is a from-scratch build of the Story Writer service described in
`README.md`. You have wide latitude over architecture and implementation. The
notes below describe what is *available* in the environment — use anything that
helps, ignore what doesn't. **Nothing here is mandatory.**

## Language
Python is preferred, but not required if you have a strong reason to choose
otherwise.

## Available tools (optional)
- **Ollama** — local LLM inference at `http://localhost:11434`. Free, offline,
  no per-token cost. A sensible default for the free tier and for iteration.
- **Hosted LLM APIs** — keys may be present in `.env` (e.g. OpenAI / DeepSeek).
  Reasonable for the paid tier or higher quality.
- **Langfuse** — LLM observability / tracing. Keys in `.env`. Wire it in if you
  want visibility into generations.
- **Sentry** — error monitoring. DSN in `.env`. Useful once a service is running.

## Other AI agents (optional collaborators)
You are not the only agent available. Where it helps, delegate:
- **codex** (or another model / CLI) is useful for *dialectical / adversarial
  review* — have it critique a design or red-team your code, then reconcile the
  feedback.
- For important decisions, prefer a second agent as a reviewer/critic over
  trusting a single pass.
Treat these as collaborators, not authorities — weigh their input.

## Scope note
Auth, payments, and tier enforcement may be **stubbed / mocked** (in-memory is
fine). The point of this build is the story-generation product and its quality,
not billing or hosting plumbing. Don't sink effort into real Stripe / infra
unless explicitly asked.

## Decisions
Record notable choices, tradeoffs, dead-ends, and surprises in
`EXPERIMENT_LOG.md` as you go. Short, dated entries are fine.
EOF

# ---------------------------------------------------------------------------
# EXPERIMENT_LOG.md — observational journal (seeded with the bootstrap entry).
# ---------------------------------------------------------------------------
cat > EXPERIMENT_LOG.md <<EOF
# Experiment Log

Short, dated entries on notable decisions, tradeoffs, dead-ends, and surprises.

## ${TODAY} — seed
Repo bootstrapped. Goal: build the Story Writer service in \`README.md\` — a web
app where a user submits a story request and receives an LLM-written story, with
free and paid tiers. No architecture, framework, evaluation, or tests are
pre-decided; those are open questions to resolve during the build. Auth /
payments / tier limits may be stubbed.
EOF

# ---------------------------------------------------------------------------
# .env.example — committed template. .env — local copy with empty placeholders.
# No real secrets are written; fill them in locally.
# ---------------------------------------------------------------------------
cat > .env.example <<'EOF'
# --- LLM providers ---
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
# Ollama runs locally; no API key required.
OLLAMA_BASE_URL=http://localhost:11434

# --- Observability ---
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# --- Error monitoring ---
SENTRY_DSN=

# --- App ---
APP_ENV=development
EOF

cp .env.example .env

cat > .gitignore <<'EOF'
# Environment / secrets
.env

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ruff_cache/

# Local scratch
.tmp/
*.log
EOF

# ---------------------------------------------------------------------------
# git init + initial commit
# ---------------------------------------------------------------------------
echo "==> Initializing git repository"
git init -q -b main
git add -A
git commit -q -m "Seed Story Writer experiment (lean: product brief + tool shelf)"

echo "==> Files created:"
git ls-files | sed 's/^/    /'
echo "    .env  (git-ignored, empty placeholders)"

# ---------------------------------------------------------------------------
# Optional: create + push a GitHub repo via gh
# ---------------------------------------------------------------------------
if [[ "${CREATE_GITHUB_REPO:-0}" == "1" ]]; then
  if command -v gh >/dev/null 2>&1; then
    VIS="${GH_VISIBILITY:-private}"
    echo "==> Creating GitHub repo ($VIS) and pushing"
    gh repo create "$(basename "$TARGET_DIR")" --"$VIS" --source=. --remote=origin --push
  else
    echo "!! CREATE_GITHUB_REPO=1 but \`gh\` is not installed; skipping remote creation." >&2
  fi
fi

# ---------------------------------------------------------------------------
# Print the kickoff instructions (intentionally NOT committed, so they don't
# pollute the build's context).
# ---------------------------------------------------------------------------
cat <<'EOF'

============================================================================
NEXT STEPS — run the experiment
============================================================================
1. (Optional) Fill in real keys in .env (Langfuse / Sentry / hosted LLMs).
2. Make sure your LLM backend is reachable (e.g. `ollama serve`).
3. Open Claude Code in this directory on a paid plan, v2.1.154+:

   Turn on multi-agent workflows + extended reasoning for the session:

       /effort ultracode

   Kick off the build (single prompt):

       Implement the Story Writer service described in README.md — a web app
       where a user submits a story request and gets back an LLM-written story,
       with free and paid tiers. Decide the architecture, models, evaluation,
       and tests yourself. Auth/payments/tier limits may be stubbed. Log key
       decisions in EXPERIMENT_LOG.md.

   Then set an autonomous completion condition so it keeps working:

       /goal The Story Writer web service runs locally via a documented
       command; a user can submit a story request through the web UI and
       receive a generated story; free vs paid tiers are enforced (free is
       limited, paid unlocks more); and the test suite passes. Stop after 40
       turns or if you hit a decision that needs my input.

(You can also prefix the kickoff prompt with `ultracode:` instead of using
/effort, e.g. `ultracode: Implement the Story Writer service...`.)
============================================================================
EOF
