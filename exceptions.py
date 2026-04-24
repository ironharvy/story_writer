"""Shared exception tuples for narrowly-scoped recoverable error handling.

These tuples are used to catch a defined set of "expected" runtime errors
across pipeline modules without resorting to bare ``except Exception``.
"""

RECOVERABLE_MODEL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
    IndexError,
)

RECOVERABLE_RUNTIME_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *RECOVERABLE_MODEL_EXCEPTIONS,
    OSError,
)
