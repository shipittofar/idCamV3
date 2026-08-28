"""User-visible status messages."""

from idcam.app.state_machine import AppState

USER_MESSAGES: dict[AppState, str] = {
    AppState.BOOT: "Starting...",
    AppState.INITIALIZING: "Initializing...",
    AppState.CAMERA_INITIALIZING: "Camera initializing...",
    AppState.CAMERA_READY: "Preparing...",
    AppState.IDLE: "Ready — Place document",
    AppState.DOCUMENT_DETECTED: "Document detected",
    AppState.WAITING_FOR_STABILITY: "Hold still...",
    AppState.FOCUSING: "Focusing...",
    AppState.FOCUS_VERIFICATION: "Verifying focus...",
    AppState.CAPTURING: "Capturing document — Please keep still",
    AppState.IMAGE_VALIDATION: "Checking image...",
    AppState.UPLOADING: "Sending securely...",
    AppState.WAITING_FOR_SERVER: "Processing...",
    AppState.PROCESSING_RESPONSE: "Processing...",
    AppState.SUCCESS: "Done",
    AppState.WAITING_FOR_REMOVAL: "Please remove document",
    AppState.RECOVERY: "Camera unavailable — Recovering",
    AppState.ERROR: "System unavailable — Please contact support",
    AppState.SHUTDOWN: "Shutting down...",
}

FAILURE_MESSAGES = {
    "focus_failed": "Could not focus. Please remove the document and try again.",
    "autofocus_hardware_failed": "Could not focus. Please remove the document and try again.",
    "autofocus_timeout": "Could not focus. Please remove the document and try again.",
    "invalid_document_roi": "Could not focus. Please remove the document and try again.",
    "focus_verification_failed": "Could not focus. Please remove the document and try again.",
    "still_capture_failed": "Capture failed. Please remove the document and try again.",
    "capture_failed": "Capture failed. Please remove the document and try again.",
    "upload_failed": "Unable to send document. Please remove it and try again.",
    "server_rejected": "Could not process document. Please contact support.",
    "image_invalid": "Image not usable. Please remove the document and try again.",
    "camera_unavailable": "Camera problem — recovering...",
    "network_unavailable": "Connection problem — Please wait",
    "storage_full": "Storage full — Please contact support",
    "stability_timeout": "Document moved too much. Please try again.",
}
