"""Compatibility shims for optional dependencies."""

try:
    from langfuse import observe
except ImportError:

    def observe():  # type: ignore[misc]
        def decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        return decorator
