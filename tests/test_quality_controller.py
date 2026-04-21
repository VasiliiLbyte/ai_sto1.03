from __future__ import annotations

from pathlib import Path

from quality_controller import CheckResult, QualityController


def test_parse_mode_supports_hybrid() -> None:
    assert QualityController._parse_mode("AUTO: a; MANUAL: b") == "HYBRID"
    assert QualityController._parse_mode("AUTO: a") == "AUTO"
    assert QualityController._parse_mode("MANUAL: b") == "MANUAL"


def test_discover_drafts_returns_missing(tmp_path: Path) -> None:
    for name in ("Черновик Регламент заявки техники.docx", "Черновик материально-технических отчетов.docx"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    qc = QualityController.__new__(QualityController)
    qc.verbose = False
    qc._log = lambda _: None

    found, missing = qc.discover_drafts(tmp_path)

    assert len(found) == 2
    assert len(missing) == 1


def test_calc_run_metrics_counts_required_and_manual() -> None:
    checks = [
        CheckResult("A", "s", "d", "error", True, "AUTO", "pass", "not_required", "pass", "ok"),
        CheckResult("B", "s", "d", "error", True, "AUTO", "fail", "not_required", "fail", "bad"),
        CheckResult("C", "s", "d", "warning", False, "HYBRID", "warn", "required", "manual", "check"),
    ]

    metrics = QualityController._calc_run_metrics(checks)

    assert metrics["required_auto_evaluated_count"] == 2
    assert metrics["required_fail_count"] == 1
    assert metrics["manual_required_count"] == 1
    assert metrics["required_auto_pass_rate"] == 50.0
