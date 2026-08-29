"""Active capture transaction context — owned exclusively by the orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TransactionPhase(str, Enum):
    """Lifecycle phase of an active capture transaction."""

    CANDIDATE = "candidate"
    STABILIZING = "stabilizing"
    FOCUSING = "focusing"
    FOCUS_VERIFIED = "focus_verified"
    CAPTURING = "capturing"
    VALIDATED = "validated"
    UPLOADING = "uploading"
    AWAITING_RESPONSE = "awaiting_response"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ActiveCaptureTransaction:
    """
    Single active capture transaction.

    NOT a queue item. At most one instance exists in the orchestrator.
    """

    capture_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document_detected_at: datetime | None = None
    focus_started_at: datetime | None = None
    focus_completed_at: datetime | None = None
    capture_started_at: datetime | None = None
    image_path: str | None = None
    image_size: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_sha256: str | None = None
    sharpness_score: float | None = None
    min_required_sharpness: float | None = None
    upload_started_at: datetime | None = None
    upload_completed_at: datetime | None = None
    server_status: int | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None
    phase: TransactionPhase = TransactionPhase.CANDIDATE
    focus_attempts: int = 0
    focus_verify_attempts: int = 0
    trigger: str = "auto"
    applied_async_ops: set[str] = field(default_factory=set)

    def try_consume_async(self, operation: str) -> bool:
        """Return True once per operation name; later completions are stale."""
        if operation in self.applied_async_ops:
            return False
        self.applied_async_ops.add(operation)
        return True

    @staticmethod
    def generate_id() -> str:
        """Collision-resistant capture identifier (not timestamp-only)."""
        return f"cap-{uuid.uuid4().hex}"

    def mark_document_detected(self) -> None:
        self.document_detected_at = datetime.now(timezone.utc)
        self.phase = TransactionPhase.CANDIDATE

    def mark_focus_started(self) -> None:
        self.focus_started_at = datetime.now(timezone.utc)
        self.phase = TransactionPhase.FOCUSING

    def mark_focus_completed(self, sharpness: float, min_required: float) -> None:
        self.focus_completed_at = datetime.now(timezone.utc)
        self.sharpness_score = sharpness
        self.min_required_sharpness = min_required
        self.phase = TransactionPhase.FOCUS_VERIFIED

    def mark_capture_started(self, path: str) -> None:
        self.capture_started_at = datetime.now(timezone.utc)
        self.image_path = path
        self.phase = TransactionPhase.CAPTURING

    def mark_validated(
        self,
        *,
        size: int,
        width: int,
        height: int,
        sha256: str,
        sharpness: float,
    ) -> None:
        self.image_size = size
        self.image_width = width
        self.image_height = height
        self.image_sha256 = sha256
        self.sharpness_score = sharpness
        self.phase = TransactionPhase.VALIDATED

    def mark_upload_started(self) -> None:
        self.upload_started_at = datetime.now(timezone.utc)
        self.phase = TransactionPhase.UPLOADING

    def mark_upload_completed(self, http_status: int) -> None:
        self.upload_completed_at = datetime.now(timezone.utc)
        self.server_status = http_status
        self.phase = TransactionPhase.AWAITING_RESPONSE

    def mark_completed(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.phase = TransactionPhase.COMPLETED

    def mark_failed(self, reason: str) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.failure_reason = reason
        self.phase = TransactionPhase.FAILED

    def to_log_fields(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "phase": self.phase.value,
            "sharpness_score": self.sharpness_score,
            "image_size": self.image_size,
            "image_sha256": self.image_sha256,
            "failure_reason": self.failure_reason,
            "trigger": self.trigger,
        }

    @property
    def is_terminal(self) -> bool:
        return self.phase in (TransactionPhase.COMPLETED, TransactionPhase.FAILED)
