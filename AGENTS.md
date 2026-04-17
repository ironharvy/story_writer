# AGENTS.md

This document provides guidance for AI agents working in this repository.

## Project Overview

Story Writer is an interactive DSPy-based story generation pipeline with optional image generation and Langfuse observability. It helps users create structured stories through an interactive ideation flow.

**Key Dependencies:** DSPy, Pydantic, Rich, python-dotenv, Langfuse

## Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt      # Core dependencies
pip install -r requirements-dev.txt  # Dev dependencies (pytest, ruff)
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

### DSPy Pipeline Optimization
```bash
python scripts/optimize_text_pipeline.py \
  --model openai/gpt-4o-mini \
  --manifest .tmp/dspy_optimized/text_pipeline_manifest.json
```

## Code Style Guidelines

### General Principles
- Use type hints for all function parameters and return values
- Keep functions focused and small (prefer < 50 lines)
- Use descriptive variable names; avoid single letters except in short loops
- Add docstrings to public functions and classes

### Imports
```python
# Standard library first, then third-party, then local
import logging
import re
from typing import List, Optional

import dspy
from pydantic import BaseModel, Field

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
