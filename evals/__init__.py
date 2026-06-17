"""LLM-judge evaluation gates for drafted chapters and whole manuscripts.

These complement the deterministic heuristics in ``qa.py`` and the POV
classifier in ``pov_check.py`` with reasoning-based checks that the cheap
string heuristics can't express:

- ``bible_consistency`` — does a chapter portray locations, characters, and
  the world's rules (e.g. a magic system's costs) the way the bible defines
  them?
- ``comprehension`` — a closed-book reading-comprehension self-review: derive
  the questions this chapter is supposed to let a reader answer, answer them
  from the prose alone, and grade the gaps.
- ``continuity`` — cross-chapter contradiction detection over the whole draft.
- ``premise_fidelity`` — does the finished manuscript actually dramatize the
  user's idea?

Every gate exposes a pure aggregation entry point plus an isolated DSPy call
so unit tests can stub the model. Findings use the shared
:class:`evals.types.EvalFinding` so the reviewed generator and the bench can
consume them uniformly.
"""

from evals.types import EvalFinding, Severity, worst_severity

__all__ = ["EvalFinding", "Severity", "worst_severity"]
