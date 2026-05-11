"""Bounded retry + empty-output guard for DSPy structured-output calls.

A truncated LM response makes the DSPy adapter fail to parse the structured
output field and hand back ``None`` (or an empty value); downstream code then
dies on ``len(None)`` / ``None.story_clock`` and the whole multi-hour run is
lost over one bad chapter. :func:`call_with_retry` reruns such a call a few
times and, if it still cannot get a usable result, raises :class:`LMOutputError`
so the caller can fail just that step -- and write a placeholder so the artifact
still renders -- instead of crashing the pipeline.
"""

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0


class LMOutputError(RuntimeError):
    """A DSPy call kept returning an unusable (empty/``None``) structured output."""


def _looks_empty(value: object) -> bool:
    """True if ``value`` is the kind of thing a truncated/failed parse leaves behind."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def call_with_retry(
    program,
    *,
    fields=(),
    label: str,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    **kwargs,
):
    """Call ``program(**kwargs)`` (a DSPy module invocation), retrying on failure.

    An attempt counts as failed if the call raises *or* if any attribute named in
    ``fields`` is empty/``None`` on the returned prediction. After ``attempts``
    failed tries, raise :class:`LMOutputError`.

    Args:
        program: The configured DSPy module (e.g. ``dspy.ChainOfThought(Sig)``).
        fields: Name, or iterable of names, of output field(s) that must be
            non-empty. Empty/omitted means "retry on exceptions only".
        label: Human-readable step name, used in log messages and the error.
        attempts: Maximum number of tries (>= 1).
        backoff_seconds: Multiplier for the linear sleep between tries.
        **kwargs: Forwarded to ``program``.
    """
    field_names = (fields,) if isinstance(fields, str) else tuple(fields)
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = program(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any LM/transport/parse error is retryable
            last_error = exc
            logger.warning(
                "LM step %r failed on attempt %d/%d: %s", label, attempt, attempts, exc
            )
        else:
            empty = [f for f in field_names if _looks_empty(getattr(result, f, None))]
            if not empty:
                if attempt > 1:
                    logger.info(
                        "LM step %r recovered on attempt %d/%d", label, attempt, attempts
                    )
                return result
            last_error = LMOutputError(
                f"{label}: empty output field(s): {', '.join(empty)}"
            )
            logger.warning(
                "LM step %r returned empty %s on attempt %d/%d "
                "(response was probably truncated; try a larger --max-tokens)",
                label, empty, attempt, attempts,
            )
        if attempt < attempts:
            time.sleep(backoff_seconds * attempt)
    raise LMOutputError(
        f"{label}: no usable output after {attempts} attempts"
    ) from last_error
