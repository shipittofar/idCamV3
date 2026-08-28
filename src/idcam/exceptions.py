"""Application exceptions."""


class IdCamError(Exception):
    """Base exception for all idCam errors."""

    recoverable: bool = False


class ConfigurationError(IdCamError):
    """Invalid or missing configuration."""

    recoverable = False


class CameraError(IdCamError):
    """Camera operation failed."""

    recoverable = True


class CameraUnavailableError(CameraError):
    """Camera is not available."""

    recoverable = True


class AutofocusError(CameraError):
    """Autofocus did not converge."""

    recoverable = True

    def __init__(self, message: str, *, reason: str = "focus_failed") -> None:
        super().__init__(message)
        self.reason = reason


class CaptureError(CameraError):
    """Image capture failed."""

    recoverable = True


class ImageValidationError(IdCamError):
    """Captured image failed validation."""

    recoverable = False


class NetworkError(IdCamError):
    """Network communication failed."""

    recoverable = True


class ServerRejectedError(NetworkError):
    """Server rejected the request (4xx)."""

    recoverable = False


class ServerError(NetworkError):
    """Server error (5xx)."""

    recoverable = True


class ProtocolError(NetworkError):
    """Server response protocol violation."""

    recoverable = False


class StorageError(IdCamError):
    """Insufficient storage for capture."""

    recoverable = True


class TransactionError(IdCamError):
    """Capture transaction failed."""

    recoverable = True


class StateMachineError(IdCamError):
    """Invalid state transition."""

    recoverable = False


class RecoveryExhaustedError(IdCamError):
    """Recovery attempts exhausted."""

    recoverable = False


class ShutdownRequested(IdCamError):
    """Graceful shutdown requested."""

    recoverable = False
