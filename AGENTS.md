# AGENTS.md

This document provides guidance for AI agents working in this repository.

## Project Overview

Story Writer is an interactive DSPy-based story generation pipeline with optional image generation and Langfuse observability. It helps users create structured stories through an interactive ideation flow.

**Key Dependencies:** DSPy, Pydantic, Rich, python-dotenv, Langfuse

## Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt      # Core dependencies
pip install -r requirements-dev.txt  # Dev dependencies (pytest, ruff, pylint, radon, vulture)
```

### Running Tests
```bash
pytest -q                              # Run all tests (quiet mode)
pytest -q test_story.py               # Run specific test file
pytest test_story.py::test_function   # Run single test function
pytest -k "test_name"                # Run tests matching pattern
pytest --tb=short                     # Short traceback format
```

### Linting (Ruff)
```bash
ruff check .                          # Check all files
ruff check . --fix                    # Auto-fix issues
ruff check file.py                    # Check specific file
```

### Formatting (Ruff)
```bash
ruff format .                         # Format all files
ruff format file.py                   # Format specific file
ruff format --check .                 # Check formatting without changes
```

### Running the Application
```bash
python main.py                        # Basic run with default model
python main.py --enable-images       # With image generation
python main.py -v                    # Verbose (INFO)
python main.py -vv                   # LLM debug logging
python main.py -vvv                  # Full HTTP+LLM firehose
```

### Code Quality Checks
```bash
pylint main.py story_modules.py        # Design smell detection
radon cc main.py -s -n C               # Cyclomatic complexity (flag C and worse)
radon mi main.py -s                     # Maintainability index
vulture . --min-confidence 80           # Dead code detection
```

### DSPy Pipeline Optimization
```bash
python scripts/optimize_text_pipeline.py \
  --model openai/gpt-4o-mini \
  --manifest .tmp/dspy_optimized/text_pipeline_manifest.json
```

## Architecture & Design Rules

These rules are **mandatory**. AI agents must follow them for all new code and
refactor violations when touching existing code.

### Function Size & Responsibility
- **Hard limit: 50 lines per function.** If a function exceeds 50 lines, it must be decomposed. No exceptions for "entry points" or "main functions."
- **One job per function.** A function that parses arguments must not also initialize services, run a pipeline, or write files. Name each function after its single responsibility.
- **CLI entry points are thin.** `main()` should parse args, call a handful of orchestration functions, and return. Target: ≤ 20 lines of logic (excluding the argparse block, which should be extracted to `build_arg_parser() -> argparse.ArgumentParser`).

### Data Flow & State
- **Use dataclasses for pipeline state.** When ≥ 3 related values are threaded through multiple functions, define a `@dataclass` to hold them. Never pass 5+ loose locals between pipeline phases.
- **Functions take explicit inputs and return explicit outputs.** Avoid relying on mutable shared state. Each pipeline phase should be a pure-ish function: `def generate_spine(...) -> SpineResult`.

### Indirection Must Earn Its Keep
- **No dict-of-instances when a dataclass works.** If you create a dict and immediately destructure it into named variables, use a dataclass or NamedTuple instead. String keys for typed objects are a code smell.
- **No wrapper functions that just forward arguments.** If a function's body is a single call to another function with the same arguments, delete the wrapper.

### Separation of Concerns
- **UI code must not live inside business logic.** `console.print()`, `Prompt.ask()`, `Confirm.ask()` belong in a UI/interaction layer. Pipeline functions must not import `rich`. Pass callbacks or return data and let the caller handle presentation.
- **File I/O gets its own function.** Any block of ≥ 5 lines writing to a file must be a dedicated `save_*()` function with a clear signature.

### Dead Code & Commented-Out Code
- **Delete commented-out code.** Version control exists. Never leave `#old_function_call()` or `field="[REMOVED]"` in the codebase. If a feature is removed, remove all traces.
- **Delete unused definitions.** If a class, function, or constant has no callers, delete it.

### Error Reporting
- **Use `logger` for all errors and warnings.** Never use `console.print("[bold red]Error:...")` for error handling. `console.print` is for user-facing output only; `logger.error/warning` is for error handling. User-visible errors should use `logger` and optionally `sys.exit(1)`.

### Module Size
- **One module, one domain.** If a file exceeds ~300 lines or contains more than one conceptual domain (e.g., Pydantic models + DSPy modules + text utilities), split it. A file named `story_modules.py` should not also be the home of generic text-cleaning utilities.

## Code Style Guidelines

### General Principles
- Use type hints for all function parameters and return values, **including return types (even `-> None`)**
- Keep functions focused and small (prefer < 50 lines)
- Use descriptive variable names; avoid single letters except in short loops
- Add docstrings to public functions and classes
- **Avoid `Any` in type hints.** If the real type is known, use it

### Imports (strict ordering — violations must be fixed before commit)
```python
# 1. Standard library
import logging
import re
from typing import List, Optional

# 2. Third-party
import dspy
from pydantic import BaseModel, Field

# 3. Local
from story_modules import QuestionGenerator
```

### Formatting
- **Line length:** Ruff default (88 characters)
- **Indentation:** 4 spaces
- **Blank lines:** Two blank lines between top-level definitions
- **Quotes:** Double quotes for strings; single quotes only for char literals
- **Trailing commas:** Required for multi-line literals (lists, dicts, calls)

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `story_modules.py` |
| Classes | PascalCase | `QuestionGenerator` |
| Functions | snake_case | `find_similar_sentences` |
| Variables | snake_case | `core_premise` |
| Constants | UPPER_SNAKE | `RANDOM_DETAIL_PROBABILITY` |
| Private | _leading_underscore | `_normalize()` |
| Type vars | PascalCase | `T = TypeVar("T")` |

### Type Annotations
```python
# Prefer specific types over Any
def process_story(text: str) -> list[str]:
    ...

def find_matches(text: str, threshold: float = 0.65) -> list[tuple[str, str, float]]:
    ...

# Use Optional for nullable params
def configure(model_name: str, api_key: str | None = None) -> None:
    ...

# Use | for union types (Python 3.10+)
output_dir: str | None = None
```

### DSPy Patterns

**Signatures:**
```python
class GenerateQuestionsSignature(dspy.Signature):
    """One-line summary of what this signature does."""
    idea: str = dspy.InputField(desc="Description of the input.")
    questions: list[QuestionType] = dspy.OutputField(desc="Description of output.")
```

**Modules:**
```python
class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQuestionsSignature)

    @observe()  # From _compat import observe
    def forward(self, idea: str):
        return self.generate(idea=idea)
```

### Pydantic Models
```python
class QuestionWithAnswer(BaseModel):
    question: str = Field(description="The interrogative question.")
    proposed_answer: str = Field(description="A proposed answer.")

    @model_validator(mode='before')
    @classmethod
    def fix_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Normalize field names here
        return normalized
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels:
logger.debug("Detailed debug info")
logger.info("Normal operation info")
logger.warning("Something unexpected but handled")
logger.error("Error occurred", exc_info=True)
```

### Error Handling
```python
# Prefer specific exceptions
try:
    result = some_function()
except ValueError as e:
    logger.warning("Invalid input: %s", e)
    return None

# Use logger.exception for errors with traceback
except Exception as e:
    logger.exception("Unexpected error in process_story: %s", e)
    raise
```

### Testing
```python
@pytest.fixture
def mock_lm_configured():
    """Configure DSPy with the mock LM for unit tests."""
    lm = MockLM()
    dspy.configure(lm=lm)
    return lm

def test_specific_function():
    result = function_under_test(input_data)
    assert result.expected_field == "value"
```

### File Organization
- `main.py` - CLI entry point and orchestration
- `story_modules.py` - Core story generation modules
- `world_bible_modules.py` - World bible generation
- `postprocessing.py` - Post-generation utilities
- `image_gen.py` - Replicate image generation
- `logging_config.py` - Centralized logging setup
- `dspy_optimization.py` - Module loading/optimization
- `test_*.py` - Test files (mirror module structure)

### Docstrings
```python
def find_similar_sentences(
    text: str,
    threshold: float = 0.65,
) -> list[tuple[str, str, float]]:
    """Find similar sentence pairs using alphabetical-sort clustering.

    Steps:
        1. Extract and normalize sentences.
        2. Sort alphabetically — similar sentences cluster together.
        3. Compare adjacent pairs with SequenceMatcher.
        4. Return pairs above the similarity threshold.

    Args:
        text: The story text to analyze.
        threshold: Minimum similarity ratio (0-1). Default 0.65.

    Returns:
        List of (sentence_a, sentence_b, similarity_score) tuples,
        sorted by score descending.
    """
```

### Regex Patterns
```python
# Compile regex at module level for reuse
_chapter_prefix_re = re.compile(
    r'^(chapter\s+\d+\s*[:\-\.]\s*)',
    re.IGNORECASE,
)

# Use raw strings for regex
_match = _chapter_prefix_re.match(title)
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL` | LLM model to use | `openai/gpt-4o-mini` |
| `LLM_URL` | Custom API endpoint | - |
| `API_KEY` | API authentication | - |
| `OPENAI_API_KEY` | OpenAI auth | - |
| `REPLICATE_API_TOKEN` | Image generation | - |
| `DSPY_CACHE_DIR` | DSPy disk cache | - |
| `LOG_LEVEL` | Log verbosity | `INFO` |
| `LOG_FORMAT` | `text` or `json` | `text` |
| `LOG_FILE` | Log file path | - |
| `LANGFUSE_*` | Observability config | - |

## Common Tasks

### Adding a New DSPy Signature
1. Define the signature class inheriting from `dspy.Signature`
2. Document inputs/outputs with `desc=` parameters
3. Create a module class inheriting from `dspy.Module`
4. Initialize `dspy.Predict` or `dspy.ChainOfThought` in `__init__`
5. Add `@observe()` decorator to the `forward` method
6. Write tests with mock LM responses

### Adding Tests
1. Create `test_<module>.py` following existing pattern
2. Use `MockLM` subclass for LLM-dependent tests
3. Use `@pytest.mark.parametrize` for multiple input/output cases
4. Test Pydantic model validation and field normalization
5. Mock external services (Replicate, Langfuse) with `unittest.mock`
