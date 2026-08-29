"""Application state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from idcam.exceptions import StateMachineError

if TYPE_CHECKING:
    from idcam.config.schema import StateTimeoutsConfig


class AppState(str, Enum):
    BOOT = "BOOT"
    INITIALIZING = "INITIALIZING"
    CAMERA_INITIALIZING = "CAMERA_INITIALIZING"
    CAMERA_READY = "CAMERA_READY"
    IDLE = "IDLE"
    DOCUMENT_DETECTED = "DOCUMENT_DETECTED"
    WAITING_FOR_STABILITY = "WAITING_FOR_STABILITY"
    FOCUSING = "FOCUSING"
    FOCUS_VERIFICATION = "FOCUS_VERIFICATION"
    CAPTURING = "CAPTURING"
    IMAGE_VALIDATION = "IMAGE_VALIDATION"
    UPLOADING = "UPLOADING"
    WAITING_FOR_SERVER = "WAITING_FOR_SERVER"
    PROCESSING_RESPONSE = "PROCESSING_RESPONSE"
    SUCCESS = "SUCCESS"
    WAITING_FOR_REMOVAL = "WAITING_FOR_REMOVAL"
    RECOVERY = "RECOVERY"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


# Active capture/upload — not IDLE and not the post-success removal hold.
CAPTURE_IN_PROGRESS_STATES = {
    AppState.DOCUMENT_DETECTED,
    AppState.WAITING_FOR_STABILITY,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.SUCCESS,
}

# States with no timeout (must not block forever via watchdog elsewhere)
NO_TIMEOUT_STATES = {
    AppState.CAMERA_READY,
    AppState.IDLE,
    AppState.ERROR,
}

ALLOWED_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.BOOT: {AppState.INITIALIZING, AppState.SHUTDOWN},
    AppState.INITIALIZING: {AppState.CAMERA_INITIALIZING, AppState.ERROR, AppState.SHUTDOWN},
    AppState.CAMERA_INITIALIZING: {AppState.CAMERA_READY, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.CAMERA_READY: {AppState.IDLE, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.IDLE: {AppState.DOCUMENT_DETECTED, AppState.FOCUSING, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.DOCUMENT_DETECTED: {AppState.WAITING_FOR_STABILITY, AppState.FOCUSING, AppState.IDLE, AppState.WAITING_FOR_REMOVAL, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.WAITING_FOR_STABILITY: {AppState.FOCUSING, AppState.IDLE, AppState.WAITING_FOR_REMOVAL, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.FOCUSING: {AppState.FOCUS_VERIFICATION, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.ERROR, AppState.SHUTDOWN},
    AppState.FOCUS_VERIFICATION: {AppState.CAPTURING, AppState.FOCUSING, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.CAPTURING: {AppState.IMAGE_VALIDATION, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.IMAGE_VALIDATION: {AppState.UPLOADING, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.UPLOADING: {AppState.WAITING_FOR_SERVER, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.WAITING_FOR_SERVER: {AppState.PROCESSING_RESPONSE, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.PROCESSING_RESPONSE: {AppState.SUCCESS, AppState.WAITING_FOR_REMOVAL, AppState.IDLE, AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.SUCCESS: {AppState.WAITING_FOR_REMOVAL, AppState.SHUTDOWN},
    AppState.WAITING_FOR_REMOVAL: {
        AppState.IDLE,
        AppState.DOCUMENT_DETECTED,
        AppState.RECOVERY,
        AppState.SHUTDOWN,
    },
    AppState.RECOVERY: {AppState.CAMERA_INITIALIZING, AppState.CAMERA_READY, AppState.IDLE, AppState.ERROR, AppState.SHUTDOWN},
    AppState.ERROR: {AppState.RECOVERY, AppState.SHUTDOWN},
    AppState.SHUTDOWN: set(),
}


@dataclass
class StateMachine:
    """Explicit deterministic state machine with configurable per-state timeouts."""

    state: AppState = AppState.BOOT
    entered_at: float = field(default_factory=time.monotonic)
    last_progress_at: float = field(default_factory=time.monotonic)
    transition_count: int = 0
    _timeouts: StateTimeoutsConfig | None = field(default=None, repr=False)

    def set_timeouts(self, timeouts: StateTimeoutsConfig) -> None:
        self._timeouts = timeouts

    def can_transition(self, target: AppState) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self.state, set())

    def transition(self, target: AppState, *, reason: str = "") -> AppState:
        if not self.can_transition(target):
            raise StateMachineError(
                f"Invalid transition {self.state.value} -> {target.value}" + (f": {reason}" if reason else "")
            )
        self.state = target
        now = time.monotonic()
        self.entered_at = now
        self.last_progress_at = now
        self.transition_count += 1
        return self.state

    def touch_progress(self) -> None:
        """Record progress within current state (resets watchdog stall detection)."""
        self.last_progress_at = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.entered_at

    def progress_idle(self) -> float:
        return time.monotonic() - self.last_progress_at

    def timeout_for_state(self, state: AppState | None = None) -> float | None:
        target = state or self.state
        if target in NO_TIMEOUT_STATES:
            return None
        if self._timeouts is not None:
            return self._timeouts.for_state(target.value)
        return None

    def is_timed_out(self, override_timeout: float | None = None) -> bool:
        timeout = override_timeout if override_timeout is not None else self.timeout_for_state()
        if timeout is None:
            return False
        return self.elapsed() > timeout

    def in_transaction(self) -> bool:
        """True while a capture/upload is running.

        WAITING_FOR_REMOVAL is a post-success hold, not an active capture.
        """
        return self.state in CAPTURE_IN_PROGRESS_STATES
