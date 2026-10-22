"""Domain-level exceptions. The HTTP layer maps each one to a status code."""


class OcrError(Exception):
    """Base class for every error raised by this service."""


class InvalidDocumentError(OcrError):
    """The uploaded file is empty, corrupt, or of an unsupported type."""


class DocumentTooLargeError(OcrError):
    """The uploaded file exceeds a configured size or page limit."""


class EngineUnavailableError(OcrError):
    """The OCR engine could not be reached."""


class EngineResponseError(OcrError):
    """The OCR engine answered with an error or an unexpected payload."""


class InvalidRequestError(OcrError):
    """A request parameter is invalid or not allowed by the server configuration."""


class JobNotFoundError(OcrError):
    """No job exists with the requested id."""


class JobStateError(OcrError):
    """The job is not in a state that allows the requested operation."""
