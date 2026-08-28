"""The camera diagnostic must keep working, and must stay independent.

These run everywhere (mock backend). The point of the diagnostic is to be
usable when the rest of the application is broken, so the import-isolation
test below is as important as the run test.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "camera_capture_test.py"


def run_diagnostic(tmp_path: Path, *extra: str) -> dict:
    summary = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mock",
            "--json-out", str(summary),
            "--fps-frames", "3",
            "--output", str(tmp_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert summary.is_file(), result.stdout + result.stderr
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["_returncode"] = result.returncode
    return payload


def test_script_exists():
    assert SCRIPT.is_file()


def test_runs_against_mock_backend_and_passes(tmp_path):
    payload = run_diagnostic(
        tmp_path,
        "--preview-width", "320",
        "--preview-height", "240",
        "--capture-width", "640",
        "--capture-height", "480",
    )
    assert payload["_returncode"] == 0
    assert payload["ok"] is True

    names = [step["name"] for step in payload["steps"]]
    for expected in (
        "initialize camera",
        "stream geometry",
        "acquire preview frame",
        "preview fps",
        "rgb channel ordering",
        "full-resolution capture",
        "validate JPEG",
        "shutdown camera",
    ):
        assert expected in names, names


def test_capture_step_uses_a_jpg_extension_for_the_encoder(tmp_path):
    """The in-progress file the encoder sees must still end in .jpg."""
    payload = run_diagnostic(tmp_path, "--capture-width", "640", "--capture-height", "480")
    capture = next(s for s in payload["steps"] if s["name"] == "full-resolution capture")
    assert capture["passed"] is True
    assert capture["detail"]["encoder_sees_extension"] == ".jpg"
    assert ".partial" in capture["detail"]["partial_path"]
    assert capture["detail"]["partial_removed"] is True


def test_reports_measured_preview_fps(tmp_path):
    payload = run_diagnostic(tmp_path, "--skip-capture")
    fps_step = next(s for s in payload["steps"] if s["name"] == "preview fps")
    assert fps_step["passed"] is True
    assert fps_step["detail"]["fps"] > 0
    assert fps_step["detail"]["frames"] == 3


def test_validates_captured_jpeg_dimensions(tmp_path):
    payload = run_diagnostic(tmp_path, "--capture-width", "800", "--capture-height", "600")
    validation = next(s for s in payload["steps"] if s["name"] == "validate JPEG")
    assert validation["passed"] is True
    assert validation["detail"]["pil_format"] == "JPEG"
    assert validation["detail"]["expected_size (w, h)"] == [800, 600]
    assert validation["detail"]["file_bytes"] > 0


@pytest.mark.parametrize(
    "forbidden",
    ["idcam.app.orchestrator", "idcam.app.state_machine", "idcam.camera.autofocus", "idcam.vision"],
)
def test_does_not_import_the_state_machine(forbidden):
    """A camera diagnostic that pulls in the pipeline cannot isolate faults."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith(forbidden) for name in imported), sorted(imported)
