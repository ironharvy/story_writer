# New Story Writer experiment — bootstrap

`bootstrap_story_writer.sh` scaffolds a **fresh, deliberately lean** repo for a
second Story Writer experiment. The thesis: *minimal seed → maximal autonomy →
observe what Claude builds on its own.*

What it intentionally **does** seed (ambient capability):
- A one-paragraph product brief (`README.md`).
- A "tool shelf" (`CLAUDE.md`) listing Ollama, hosted LLMs, Langfuse, Sentry,
  and *other AI agents* (e.g. codex for dialectical/adversarial review) — all
  marked **optional**.
- `.env` / `.env.example` with placeholder keys.
- An `EXPERIMENT_LOG.md` decision journal.

What it intentionally **does not** seed (observed behavior): architecture,
web framework, model choice, story evaluation, tests. The point is to see
whether Claude decides those matter and builds them. (Auth/payments/tiers are
explicitly allowed to be stubbed.)

## Usage

```bash
./bootstrap_story_writer.sh my-story-writer        # scaffold into ./my-story-writer

# Optionally also create + push a GitHub repo (requires gh):
CREATE_GITHUB_REPO=1 GH_VISIBILITY=private ./bootstrap_story_writer.sh my-story-writer
```

The script prints the `/effort ultracode` kickoff prompt and the `/goal`
completion condition to paste into Claude Code once the repo exists. Those
instructions are printed, **not committed**, so they don't pollute the build's
context.
