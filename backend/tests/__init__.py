"""
Minimal backend package initializer.

Provides lazy access to CreateSheetMusic (avoids importing heavy deps at package import time).
Add other exports to __all__ and __getattr__ as needed.
"""
__all__ = ["CreateSheetMusic"]


def __getattr__(name: str):
    if name == "CreateSheetMusic":
        from .main import CreateSheetMusic  # imported on first access
        return CreateSheetMusic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
