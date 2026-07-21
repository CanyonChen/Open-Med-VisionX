"""Typed, user-facing exceptions used across platform layers."""


class ViewerError(Exception):
    """Base class for recoverable platform errors."""


class ValidationError(ViewerError, ValueError):
    """Input violates a public interface contract."""


class UnsupportedFormatError(ViewerError):
    """No registered loader accepts the input."""


class FormatMismatchError(ViewerError):
    """A file extension and its decoded signature disagree."""


class DecodeError(ViewerError):
    """A supported file is corrupt, truncated, or cannot be decoded."""


class ResourceLimitError(ViewerError):
    """An input exceeds a configured safety limit."""


class MissingDependencyError(ViewerError, ImportError):
    """An optional feature was requested without its dependency installed."""


class OperationCancelled(ViewerError):
    """A cooperative background operation was cancelled."""


class PluginError(ViewerError):
    """An external model or loader plugin failed safely."""


class ProviderError(ViewerError):
    """A cloud provider request failed without exposing credentials."""
