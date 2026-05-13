class PipelineError(Exception):
    """Base pipeline error."""


class DataValidationError(PipelineError):
    """Raised when a dataset fails validation."""


class FetchError(PipelineError):
    """Raised when an upstream data provider fails."""
