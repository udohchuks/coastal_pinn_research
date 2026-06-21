"""Domain-specific exceptions raised by coastal_pinn."""

from __future__ import annotations


class CoastalPINNError(Exception):
    """Base class for all coastal_pinn errors."""


class ConfigError(CoastalPINNError):
    """Invalid or missing configuration (YAML schema, required fields, etc.)."""


class SourceUnavailable(CoastalPINNError):
    """A data source could not be reached or returned no usable data.

    Carries the source name and the underlying cause so the CLI can render
    an actionable error message.
    """

    def __init__(self, source: str, message: str, *, cause: Exception | None = None):
        self.source = source
        self.cause = cause
        super().__init__(f"[{source}] {message}" + (f" (cause: {cause!r})" if cause else ""))


class MissingCredentials(CoastalPINNError):
    """Required credentials (e.g., Copernicus Marine) were not found."""


class SchemaError(CoastalPINNError):
    """A produced DataFrame does not match the documented schema."""


class AppendOnlyCacheMiss(CoastalPINNError):
    """Internal: a cache lookup found no matching artifact (not an error for callers).

    fetch_* functions may raise this internally to signal "fetch from network".
    The pipeline translates it to either a cache hit or a real fetch.
    """