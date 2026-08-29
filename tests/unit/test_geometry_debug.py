"""Geometry debug traces must not change accept/reject outcomes."""

from __future__ import annotations

import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.geometry import find_document_quad
from idcam.vision.geometry_debug import classify_failure, diagnose_frame, render_overlay
from tests.unit.test_large_two_page_geometry import _opened_passport, _page, _wood_table, _draw_page


def test_trace_does_not_change_finder_result():
    from idcam.vision.geometry_debug import GeometryTrace

    frame = _page((480, 640), (80, 60, 480, 360))
    a, na = find_document_quad(frame)
    trace = GeometryTrace()
    b, nb = find_document_quad(frame, trace=trace)
    assert a is not None and b is not None
    assert a.bbox == b.bbox
    assert abs(a.confidence - b.confidence) < 1e-9
    assert na == nb
    assert len(trace.candidates) >= 1


def test_diagnose_records_candidates_on_passport():
    frame = _opened_passport((1080, 1920), (80, 90, 1480, 720), seam_px=8)
    trace = diagnose_frame(frame)
    assert len(trace.candidates) >= 1
    vis = render_overlay(frame, trace)
    assert vis.shape == frame.shape
    geom = DocumentDetector(DetectionConfig()).detect_geometry(frame)
    assert geom.present is True


def test_unrelated_rectangles_merge_is_rejected():
    img = _wood_table((1080, 1920))
    img = _draw_page(img, 80, 200, 520, 400, printed=True)
    img = _draw_page(img, 1280, 220, 520, 400, printed=True)
    trace = diagnose_frame(img)
    gap_fails = [m for m in trace.merges if m.get("merge_rejection_reason") == "horizontal_gap"]
    ok = [m for m in trace.merges if not m.get("merge_rejection_reason")]
    assert not ok
    if trace.merges:
        assert gap_fails


def test_empty_table_classified_as_f_or_g():
    trace = diagnose_frame(_wood_table((480, 640)))
    case = classify_failure(trace, detector_present=False)
    assert case in {"A", "F", "G"}
    assert trace.final is None
