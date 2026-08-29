"""State machine unit tests."""

import time

import pytest

from idcam.app.state_machine import AppState, StateMachine
from idcam.config.schema import StateTimeoutsConfig
from idcam.exceptions import StateMachineError


@pytest.fixture
def timeouts():
    return StateTimeoutsConfig(
        document_detected=5.0,
        focusing=10.0,
        waiting_for_stability=15.0,
    )


class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.state == AppState.BOOT

    def test_valid_transition(self):
        sm = StateMachine()
        sm.transition(AppState.INITIALIZING)
        assert sm.state == AppState.INITIALIZING

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(StateMachineError):
            sm.transition(AppState.CAPTURING)

    def test_idle_to_document_detected(self):
        sm = StateMachine(state=AppState.IDLE)
        sm.transition(AppState.DOCUMENT_DETECTED)
        assert sm.in_transaction()

    def test_full_happy_path_transitions(self):
        sm = StateMachine(state=AppState.IDLE)
        for target in [
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
            AppState.WAITING_FOR_REMOVAL,
            AppState.IDLE,
        ]:
            sm.transition(target)

    def test_manual_skips_stability(self):
        sm = StateMachine(state=AppState.IDLE)
        sm.transition(AppState.DOCUMENT_DETECTED)
        sm.transition(AppState.FOCUSING)
        assert sm.state == AppState.FOCUSING

    def test_idle_can_go_directly_to_focusing(self):
        sm = StateMachine(state=AppState.IDLE)
        sm.transition(AppState.FOCUSING)
        assert sm.state == AppState.FOCUSING

    def test_recovery_from_focusing(self):
        sm = StateMachine(state=AppState.FOCUSING)
        sm.transition(AppState.RECOVERY)
        assert sm.state == AppState.RECOVERY

    def test_timeout_detection(self, timeouts):
        sm = StateMachine(state=AppState.DOCUMENT_DETECTED)
        sm.set_timeouts(timeouts)
        sm.entered_at = time.monotonic() - 100
        assert sm.is_timed_out()

    def test_no_timeout_for_idle(self, timeouts):
        sm = StateMachine(state=AppState.IDLE)
        sm.set_timeouts(timeouts)
        sm.entered_at = time.monotonic() - 10000
        assert not sm.is_timed_out()

    def test_timeout_for_removal(self, timeouts):
        sm = StateMachine(state=AppState.WAITING_FOR_REMOVAL)
        sm.set_timeouts(timeouts)
        assert sm.timeout_for_state() == timeouts.waiting_for_removal
        sm.entered_at = time.monotonic() - 10000
        assert sm.is_timed_out()

    def test_in_transaction_states(self):
        sm = StateMachine(state=AppState.UPLOADING)
        assert sm.in_transaction()
        sm = StateMachine(state=AppState.IDLE)
        assert not sm.in_transaction()
        sm = StateMachine(state=AppState.WAITING_FOR_REMOVAL)
        assert not sm.in_transaction()

    def test_touch_progress_does_not_reset_state_elapsed(self, timeouts):
        sm = StateMachine(state=AppState.CAPTURING)
        sm.set_timeouts(timeouts)
        sm.entered_at = time.monotonic() - 100
        sm.last_progress_at = time.monotonic()
        # State timeout uses entered_at; progress is for watchdog
        assert sm.is_timed_out()
        assert sm.progress_idle() < 1.0

    def test_per_state_timeout_from_config(self, timeouts):
        sm = StateMachine(state=AppState.FOCUSING)
        sm.set_timeouts(timeouts)
        assert sm.timeout_for_state() == 10.0
