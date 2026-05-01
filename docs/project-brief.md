# Project Brief

## Summary

Story Writer is a local-first, CLI-driven story generation pipeline. It turns a single user idea into a coherent short story, novella, or book-length plan by guiding the user through premise clarification, typed planning artifacts, chapter drafting, and QA detection.

## Design Concept

The product behaves like a disciplined writing room around a local model: it asks only the questions needed to define scope, locks key story facts into reviewable artifacts, then writes chapters from those artifacts instead of relying on conversational memory. The MVP succeeds when the generated prose is interesting, coherent, and free of obvious LLM artifacts.

## Target Users

- Writers who want a structured AI collaborator for first drafts.
- Local-model users who do not want a hosted web service or paid API by default.
- Developers experimenting with DSPy as a typed creative-writing pipeline.

## MVP Scope

- CLI only.
- Local Ollama provider by default.
- User review gates for premise, spine, and world bible.
- Persisted run artifacts under `runs/<slug>/`.
- Detection-first QA for artifacts, continuity, alignment, and repetition.

## Out of Scope

- Web UI, auth, accounts, story sharing, images, audio, telemetry, database storage, Docker deployment, and paid providers by default.

## Success Criteria

- A user can run `story-writer new --idea "..."`
- The system asks clarifying questions with suggested answers.
- The system persists premise, spine, world bible, chapter plan, per-chapter JSON, QA reports, and rendered prose.
- The world bible is the source of truth for facts used in later stages.
- Unit tests pass without Ollama.
- A real Ollama smoke path can be run when local models are available.
