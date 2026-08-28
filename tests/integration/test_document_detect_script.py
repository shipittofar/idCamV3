"""document_detect_test.py must stay independent of the capture pipeline."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "document_detect_test.py"
_BLOCKED = (
    "idcam.app.orchestrator",
    "idcam.app.state_machine",
    "idcam.camera.autofocus",
    "idcam.camera.capture",
)


def test_script_exists():
    assert SCRIPT.is_file()


def test_does_not_import_pipeline():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(any(name.startswith(b) for b in _BLOCKED) for name in imported), sorted(imported)


def test_offline_image_runs(tmp_path):
    h, w = 480, 640
    yy, xx = np.indices((h, w))
    wood = (85 + (xx * 3 + yy * 5) % 45).astype(np.uint8)
    frame = np.stack([wood, (wood * 0.7).astype(np.uint8), (wood * 0.45).astype(np.uint8)], axis=-1)
    frame[60:420, 80:560] = (242, 238, 230)
    frame[60:63, 80:560] = 35
    frame[417:420, 80:560] = 35
    frame[60:420, 80:83] = 35
    frame[60:420, 557:560] = 35
    src = tmp_path / "page.png"
    cv2.imwrite(str(src), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    summary = tmp_path / "out.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(src),
            "--output",
            str(tmp_path),
            "--save-overlay",
            "--json-out",
            str(summary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["frames"][0]["present"] is True
    assert payload["frames"][0]["corners"] is not None
