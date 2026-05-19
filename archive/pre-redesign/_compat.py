"""
Compatibility shims for optional dependencies.
"""

try:
    from langfuse import observe
except ImportError:

    def observe():
        def decorator(fn):
            return fn

        return decorator
