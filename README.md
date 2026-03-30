# story_writer

`story_writer` is an AI-based story authoring tool. Its primary purpose is to turn a user's initial idea into an interesting, coherent story through a guided workflow. The system asks a small number of targeted questions, uses those answers to sharpen the narrative, and then generates the story. It can also optionally generate accompanying images that stay coherent with the written output.

This is not a manga generator and not an image-first tool. The product is story-first: the text is the main artifact, and images are supporting material.

## Product Definition

The project is intended to:

- Take a rough story idea from the user.
- Ask a limited number of useful follow-up questions.
- Build narrative context from the idea plus answers.
- Generate a structured story and final prose.
- Support future "knobs" to let the user steer style, scope, tone, and similar controls.
- Optionally generate coherent images for characters and scenes.

## Core Workflow

The expected workflow is:

1. The user provides an initial story idea.
2. The system generates a few clarifying questions.
3. The user accepts or edits the proposed answers.
4. The system synthesizes a core premise.
5. The system expands that into supporting narrative context.
6. The system generates story structure and final story text.
7. The system optionally generates supporting images based on the resulting story artifacts.

## Scope

In scope:

- Story ideation and refinement.
- Guided question-and-answer narrative development.
- World building and story structure generation.
- Long-form story generation.
- Optional image generation for characters and scenes.
- Backend flexibility across local and hosted model providers.

Out of scope:

- Manga or comic panel sequencing.
- Image-only generation workflows.
- Full publishing, layout, or book-production tooling.

## Model and Backend Strategy

The project should support multiple model backends behind a consistent application layer.

- Ollama is the default local backend for development and testing.
- Hosted text providers such as DeepSeek or other OpenAI-compatible backends can be used in production.
- Replicate is an optional backend for models that are not run locally, especially image models.

The application should treat model providers as interchangeable execution backends, not as product-defining architecture.

## Architecture Direction

The codebase should evolve around four clear layers:

### 1. Story Pipeline

Responsible for:

- user idea intake
- follow-up questions
- premise generation
- world bible generation
- story planning
- final story writing
- future narrative knobs

This is the core of the product and should remain independent from image-specific concerns.

### 2. Image Pipeline

Responsible for:

- character visual extraction
- portrait generation
- scene illustration generation
- keeping images consistent with story artifacts

This pipeline should be optional and should consume structured outputs from the story pipeline.

### 3. Model Backend Layer

Responsible for:

- text model configuration
- image model configuration
- local vs hosted backend selection
- provider-specific request details

This layer should hide provider-specific differences so the application can swap Ollama, DeepSeek, Replicate, or similar providers without changing product logic.

### 4. Output and Rendering Layer

Responsible for:

- terminal interaction
- markdown export
- file layout for generated assets
- future UI or API surfaces

This layer should format and present outputs, not own generation logic.

## Design Principles

- Story first: text generation is the main product value.
- Images are optional: image features should not break text-only use cases.
- Structured interfaces: modules should exchange structured data where possible instead of reparsing formatted strings.
- Backend portability: provider changes should not require rewiring the product flow.
- Testable boundaries: orchestration, generation, and rendering should be separable enough to test independently.

## Near-Term Direction

The next iterations should focus on:

- clarifying module boundaries between story generation and image generation
- introducing user-facing generation knobs
- improving structured outputs between pipeline stages
- making provider configuration cleaner
- strengthening tests around text-only and image-enabled flows
