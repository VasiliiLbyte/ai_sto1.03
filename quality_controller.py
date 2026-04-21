"""Stage 7 Quality Controller for STOFabric.

Runs end-to-end pipeline on one or multiple drafts, evaluates checklist rules,
and writes JSON + Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from controller import PipelineContext, STOFabricPipeline

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


DEFAULT_DRAFTS_DIR = Path(r"C:\Users\User\Desktop\Рабочие документы\СТО")
DEFAULT_OUTPUT_DIR = Path("_tmp_extract/quality_report")
DEFAULT_DRAFTS = [
    "Черновик Регламент заявки техники.docx",
    "Черновик материально-технических отчетов.docx",
    "Черновик Управление персоналом ЮЕА.docx",
]


@dataclass(slots=True)
class CheckResult:
    check_id: str
    section_name: str
    description: str
    severity: str
    required: bool
    mode: str  # AUTO|MANUAL|HYBRID
    auto_status: str  # pass|fail|warn|not_evaluated
    manual_status: str  # required|not_required
    final_status: str  # pass|fail|warn|manual
    evidence: str


class QualityController:
    """Quality controller for STOFabric full-pipeline validation."""

    def __init__(
        self,
        rules_path: Path,
        schema_path: Path,
        checklist_path: Path,
        output_dir: Path,
        model_key: str = "70b",
        verbose: bool = False,
    ) -> None:
        self.rules_path = rules_path
        self.schema_path = schema_path
        self.checklist_path = checklist_path
        self.output_dir = output_dir
        self.model_key = model_key
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_inputs()
        self.checklist = self._load_checklist()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _load_checklist(self) -> dict[str, Any]:
        data = yaml.safe_load(self.checklist_path.read_text(encoding="utf-8"))
        return data.get("sto_validation_checklist", {})

    def _validate_inputs(self) -> None:
        required_paths = [
            (self.rules_path, "Rules YAML"),
            (self.schema_path, "Schema JSON"),
            (self.checklist_path, "Validation checklist YAML"),
        ]
        for path, label in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def discover_drafts(self, drafts_dir: Path = DEFAULT_DRAFTS_DIR) -> tuple[list[Path], list[Path]]:
        drafts: list[Path] = []
        missing: list[Path] = []
        for name in DEFAULT_DRAFTS:
            path = drafts_dir / name
            if path.exists():
                drafts.append(path)
            else:
                missing.append(path)
                self._log(f"[WARN] Draft not found: {path}")
        return drafts, missing

    @staticmethod
    def _safe_name(path: Path) -> str:
        base = path.stem.lower()
        base = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "_", base)
        return base.strip("_") or "draft"

    def run_pipeline_for_draft(self, draft_path: Path) -> dict[str, Any]:
        safe = self._safe_name(draft_path)
        draft_out_dir = self.output_dir / safe
        draft_out_dir.mkdir(parents=True, exist_ok=True)
        output_docx = draft_out_dir / "final_sto.docx"

        ctx = PipelineContext(
            input_docx=draft_path,
            output_docx=output_docx,
            rules_path=self.rules_path,
            schema_path=self.schema_path,
            mode="full",
            model_key=self.model_key,  # type: ignore[arg-type]
            save_intermediate=True,
            verbose=self.verbose,
            dry_run=False,
            output_dir=draft_out_dir,
            mapper_json_path=draft_out_dir / "mapper.json",
            transformed_json_path=draft_out_dir / "transformed.json",
        )
        pipeline = STOFabricPipeline(ctx)
        return pipeline.run()

    @staticmethod
    def _parse_mode(check_expr: str) -> str:
        has_auto = "AUTO:" in check_expr
        has_manual = "MANUAL:" in check_expr
        if has_auto and has_manual:
            return "HYBRID"
        if has_auto:
            return "AUTO"
        return "MANUAL"

    def _evaluate_auto_status(self, check_id: str, pipeline_result: dict[str, Any]) -> tuple[str, str]:
        # AUTO heuristics (v1)
        warnings = pipeline_result.get("warnings", [])
        errors = pipeline_result.get("errors", [])
        stages = pipeline_result.get("stages", [])
        artifacts = pipeline_result.get("artifacts", {})
        status = "pass"
        evidence = "Heuristic auto-check passed"

        if pipeline_result.get("status") != "ok":
            status = "fail"
            evidence = "Pipeline status is failed"
        elif check_id in ("MAIN-01", "AR-01", "NR-01", "TERM-01", "REP-01", "RESP-01"):
            mapper_stage = next((s for s in stages if s.get("name") == "semantic_mapper"), {})
            if mapper_stage.get("summary", {}).get("errors", 0) > 0:
                status = "warn"
                evidence = "Semantic mapper reported errors; section presence uncertain"
        elif check_id in ("CHG-01", "ACK-01", "AGR-01"):
            transformed_path = artifacts.get("transformed_json")
            if transformed_path and Path(transformed_path).exists():
                transformed = json.loads(Path(transformed_path).read_text(encoding="utf-8"))
                sheets = (
                    transformed.get("sto_document_json", {})
                    .get("meta", {})
                    .get("extra_attributes", {})
                    .get("service_sheets", {})
                )
                map_key = {"CHG-01": "change_log", "ACK-01": "acquaintance", "AGR-01": "approval"}[check_id]
                if not sheets.get(map_key):
                    status = "warn"
                    evidence = f"service_sheets.{map_key} is empty"
            else:
                status = "warn"
                evidence = "transformed.json unavailable for service sheet check"
        elif check_id in ("TOC-01", "TOC-02", "TOC-03", "TOC-04"):
            docx_path = artifacts.get("output_docx")
            if not docx_path or not Path(docx_path).exists():
                status = "warn"
                evidence = "Final DOCX unavailable for TOC inspection"
        elif check_id.startswith("FMT-") and errors:
            status = "warn"
            evidence = "Pipeline contains errors; formatting conformance uncertain"

        if status == "pass" and any(check_id in w for w in warnings):
            status = "warn"
            evidence = "Related warning found in pipeline warnings"
        return status, evidence

    def _pipeline_check_result(self, item: dict[str, Any], pipeline_result: dict[str, Any], section_name: str) -> CheckResult:
        check_expr = item.get("check", "")
        mode = self._parse_mode(check_expr)
        check_id = item.get("id", "UNKNOWN")
        description = item.get("description", "")
        severity = item.get("severity", "info")
        required = bool(item.get("required", False))

        auto_status = "not_evaluated"
        evidence = "Manual validation required by checklist"
        if mode in {"AUTO", "HYBRID"}:
            auto_status, evidence = self._evaluate_auto_status(check_id, pipeline_result)

        manual_status = "required" if mode in {"MANUAL", "HYBRID"} else "not_required"

        if mode == "AUTO":
            final_status = auto_status
        elif mode == "MANUAL":
            final_status = "manual"
        else:
            # HYBRID remains manual because it still requires human validation.
            final_status = auto_status if auto_status == "fail" else "manual"

        if final_status not in {"pass", "fail", "warn", "manual"}:
            final_status = "warn"

        if mode == "MANUAL":
            return CheckResult(
                check_id=check_id,
                section_name=section_name,
                description=description,
                severity=severity,
                required=required,
                mode=mode,
                auto_status=auto_status,
                manual_status=manual_status,
                final_status=final_status,
                evidence=evidence,
            )

        return CheckResult(
            check_id=check_id,
            section_name=section_name,
            description=description,
            severity=severity,
            required=required,
            mode=mode,
            auto_status=auto_status,
            manual_status=manual_status,
            final_status=final_status,
            evidence=evidence,
        )

    def evaluate_checks(self, pipeline_result: dict[str, Any]) -> list[CheckResult]:
        results: list[CheckResult] = []
        for section in self.checklist.get("sections", []):
            section_name = section.get("name", "Unknown")
            for item in section.get("items", []):
                results.append(self._pipeline_check_result(item, pipeline_result, section_name))
        return results

    @staticmethod
    def _aggregate_statuses(checks: list[CheckResult]) -> dict[str, int]:
        agg = {"pass": 0, "fail": 0, "warn": 0, "manual": 0}
        for c in checks:
            agg[c.final_status] = agg.get(c.final_status, 0) + 1
        return agg

    @staticmethod
    def _calc_run_metrics(checks: list[CheckResult]) -> dict[str, Any]:
        auto_checks = [c for c in checks if c.mode in {"AUTO", "HYBRID"}]
        auto_evaluated = [c for c in auto_checks if c.auto_status in {"pass", "fail", "warn"}]
        required_auto_evaluated = [c for c in auto_evaluated if c.required]
        auto_pass = sum(1 for c in auto_evaluated if c.auto_status == "pass")
        required_auto_pass = sum(1 for c in required_auto_evaluated if c.auto_status == "pass")
        manual_required = sum(1 for c in checks if c.manual_status == "required")
        required_fail_count = sum(1 for c in checks if c.required and c.final_status == "fail")
        required_warn_count = sum(1 for c in checks if c.required and c.final_status == "warn")

        auto_pass_rate = round((auto_pass / len(auto_evaluated) * 100), 1) if auto_evaluated else 0.0
        required_auto_pass_rate = (
            round((required_auto_pass / len(required_auto_evaluated) * 100), 1) if required_auto_evaluated else 0.0
        )
        return {
            "auto_evaluated_count": len(auto_evaluated),
            "required_auto_evaluated_count": len(required_auto_evaluated),
            "auto_pass_rate": auto_pass_rate,
            "required_auto_pass_rate": required_auto_pass_rate,
            "manual_required_count": manual_required,
            "required_fail_count": required_fail_count,
            "required_warn_count": required_warn_count,
        }

    @staticmethod
    def _extract_issue_lists(checks: list[CheckResult]) -> dict[str, list[dict[str, Any]]]:
        def compact(c: CheckResult) -> dict[str, Any]:
            return {
                "id": c.check_id,
                "section_name": c.section_name,
                "severity": c.severity,
                "required": c.required,
                "description": c.description,
                "evidence": c.evidence,
                "mode": c.mode,
            }

        failed_items = [compact(c) for c in checks if c.final_status == "fail"]
        warning_items = [compact(c) for c in checks if c.final_status == "warn"]
        manual_items = [compact(c) for c in checks if c.manual_status == "required"]
        return {
            "failed_items": failed_items,
            "warning_items": warning_items,
            "manual_items": manual_items,
        }

    @staticmethod
    def _artifacts_with_existence(artifacts: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in artifacts.items():
            if isinstance(value, str) and value:
                out[key] = {"path": value, "exists": Path(value).exists()}
            else:
                out[key] = {"path": value, "exists": False}
        return out

    def build_json_report(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        aggregate = {"pass": 0, "fail": 0, "warn": 0, "manual": 0}
        aggregate_manual_required = 0
        aggregate_required_auto_evaluated = 0
        aggregate_required_auto_pass = 0
        for run in runs:
            for k, v in run["check_summary"].items():
                aggregate[k] += v
            m = run.get("compliance_metrics", {})
            aggregate_manual_required += int(m.get("manual_required_count", 0))
            aggregate_required_auto_evaluated += int(m.get("required_auto_evaluated_count", 0))
            aggregate_required_auto_pass += int(
                round((float(m.get("required_auto_pass_rate", 0.0)) / 100) * max(int(m.get("required_auto_evaluated_count", 0)), 0))
            )
        aggregate_required_auto_pass_rate = (
            round((aggregate_required_auto_pass / aggregate_required_auto_evaluated) * 100, 1)
            if aggregate_required_auto_evaluated
            else 0.0
        )
        return {
            "generated_at_epoch": int(time.time()),
            "runs_count": len(runs),
            "runs": runs,
            "aggregate_checks": aggregate,
            "aggregate_metrics": {
                "manual_required_count": aggregate_manual_required,
                "required_auto_pass_rate": aggregate_required_auto_pass_rate,
            },
        }

    def build_markdown_report(self, json_report: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("# STOFabric Quality Report")
        lines.append("")
        lines.append("## Aggregate Summary")
        agg = json_report.get("aggregate_checks", {})
        am = json_report.get("aggregate_metrics", {})
        lines.append("| pass | fail | warn | manual | required_auto_pass_rate | manual_required |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {agg.get('pass', 0)} | {agg.get('fail', 0)} | {agg.get('warn', 0)} | {agg.get('manual', 0)} | {am.get('required_auto_pass_rate', 0.0)}% | {am.get('manual_required_count', 0)} |"
        )
        lines.append("")

        for run in json_report.get("runs", []):
            metrics = run.get("compliance_metrics", {})
            lines.append(f"## Draft: `{run.get('draft_name')}`")
            lines.append(f"- pipeline_status: `{run.get('pipeline_status')}`")
            output_artifact = run.get("artifacts", {}).get("output_docx", {})
            lines.append(f"- output_docx: `{output_artifact.get('path')}` (exists={output_artifact.get('exists')})")
            cs = run.get("check_summary", {})
            lines.append("| pass | fail | warn | manual | auto_pass_rate | required_auto_pass_rate |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {cs.get('pass', 0)} | {cs.get('fail', 0)} | {cs.get('warn', 0)} | {cs.get('manual', 0)} | {metrics.get('auto_pass_rate', 0.0)}% | {metrics.get('required_auto_pass_rate', 0.0)}% |"
            )
            lines.append("")
            lines.append("### Gate Decision")
            gate = run.get("gate_decision", {})
            lines.append(f"- ready: `{gate.get('ready')}`")
            for reason in gate.get("reasons", []):
                lines.append(f"- reason: {reason}")
            lines.append("")
            lines.append("### Required Failed Items")
            lines.append("| id | severity | description | evidence |")
            lines.append("|---|---|---|---|")
            failed_required = [i for i in run.get("failed_items", []) if i.get("required")]
            if not failed_required:
                lines.append("| - | - | none | - |")
            else:
                for item in failed_required:
                    lines.append(f"| {item['id']} | {item['severity']} | {item['description']} | {item['evidence']} |")
            lines.append("")
            lines.append("### Manual Review Required")
            manual_items = run.get("manual_items", [])
            if not manual_items:
                lines.append("- none")
            else:
                for item in manual_items:
                    lines.append(f"- `{item['id']}` {item['description']}")
            lines.append("")
        return "\n".join(lines)

    def save_outputs(self, json_report: dict[str, Any], markdown_report: str) -> dict[str, str]:
        json_path = self.output_dir / "quality_report.json"
        md_path = self.output_dir / "quality_report.md"
        json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(markdown_report, encoding="utf-8")
        return {"json_report": str(json_path), "markdown_report": str(md_path)}

    def run_for_drafts(self, drafts: list[Path]) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for draft in drafts:
            self._log(f"[INFO] Processing draft: {draft}")
            try:
                pipeline_result = self.run_pipeline_for_draft(draft)
                checks = self.evaluate_checks(pipeline_result)
                check_summary = self._aggregate_statuses(checks)
                metrics = self._calc_run_metrics(checks)
                issues = self._extract_issue_lists(checks)
                gate_reasons: list[str] = []
                if pipeline_result.get("status") != "ok":
                    gate_reasons.append("pipeline_status_failed")
                if metrics.get("required_fail_count", 0) > 0:
                    gate_reasons.append("required_checks_failed")
                gate_ready = len(gate_reasons) == 0
                runs.append(
                    {
                        "draft_name": draft.name,
                        "draft_path": str(draft),
                        "pipeline_status": pipeline_result.get("status"),
                        "pipeline_elapsed_ms": pipeline_result.get("elapsed_ms"),
                        "pipeline_warnings_count": pipeline_result.get("warnings_count"),
                        "pipeline_errors_count": pipeline_result.get("errors_count"),
                        "artifacts": self._artifacts_with_existence(pipeline_result.get("artifacts", {})),
                        "checks": [asdict(c) for c in checks],
                        "check_summary": check_summary,
                        "compliance_metrics": metrics,
                        "failed_items": issues["failed_items"],
                        "warning_items": issues["warning_items"],
                        "manual_items": issues["manual_items"],
                        "gate_decision": {"ready": gate_ready, "reasons": gate_reasons},
                    }
                )
            except Exception as exc:  # pragma: no cover
                runs.append(
                    {
                        "draft_name": draft.name,
                        "draft_path": str(draft),
                        "pipeline_status": "failed",
                        "pipeline_elapsed_ms": 0,
                        "pipeline_warnings_count": 0,
                        "pipeline_errors_count": 1,
                        "artifacts": {},
                        "checks": [],
                        "check_summary": {"pass": 0, "fail": 1, "warn": 0, "manual": 0},
                        "compliance_metrics": {
                            "auto_evaluated_count": 0,
                            "required_auto_evaluated_count": 0,
                            "auto_pass_rate": 0.0,
                            "required_auto_pass_rate": 0.0,
                            "manual_required_count": 0,
                            "required_fail_count": 1,
                            "required_warn_count": 0,
                        },
                        "failed_items": [],
                        "warning_items": [],
                        "manual_items": [],
                        "gate_decision": {"ready": False, "reasons": ["pipeline_exception"]},
                        "failure": str(exc),
                    }
                )
        json_report = self.build_json_report(runs)
        markdown_report = self.build_markdown_report(json_report)
        outputs = self.save_outputs(json_report, markdown_report)
        return {"report": json_report, "output_files": outputs}


def _build_cli() -> argparse.ArgumentParser:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="STOFabric quality controller")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--run-all", action="store_true", help="Run quality checks on all 3 default drafts")
    mode_group.add_argument("--test", type=str, default=None, help="Run quality checks on a single draft (name or full path)")
    parser.add_argument("--drafts-dir", type=Path, default=None, help="Directory containing three default draft files")
    parser.add_argument("--rules", type=Path, default=None, help="Path to mapping-rules.yaml")
    parser.add_argument("--schema", type=Path, default=None, help="Path to schema JSON")
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Path to validation checklist YAML",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for quality reports")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["8b", "70b", "405b"],
        help="Model key for pipeline controller (8b=flash, 70b=pro, 405b=legacy alias)",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose output")
    parser.set_defaults(verbose=None)
    return parser


def _resolve_single_test(value: str, drafts_dir: Path = DEFAULT_DRAFTS_DIR) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    by_name = drafts_dir / value
    if by_name.exists():
        return by_name
    raise FileNotFoundError(f"Draft not found by path or name: {value}")


def main() -> None:
    args = _build_cli().parse_args()
    rules_path = args.rules or Path(os.getenv("STO_RULES_PATH", "mapping-rules.yaml"))
    schema_path = args.schema or Path(os.getenv("STO_SCHEMA_PATH", "sto-model.schema.json"))
    checklist_path = args.checklist or Path(os.getenv("STO_CHECKLIST_PATH", "sto-validation-checklist.yaml"))
    output_dir = args.output_dir or Path(os.getenv("STO_QUALITY_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    model_key = args.model or os.getenv("STO_MODEL_KEY", "70b")
    verbose = args.verbose if args.verbose is not None else os.getenv("STO_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    qc = QualityController(
        rules_path=rules_path,
        schema_path=schema_path,
        checklist_path=checklist_path,
        output_dir=output_dir,
        model_key=model_key,
        verbose=verbose,
    )

    drafts: list[Path]
    drafts_dir = args.drafts_dir or Path(os.getenv("STO_QUALITY_DRAFTS_DIR", str(DEFAULT_DRAFTS_DIR)))
    if args.run_all:
        drafts, missing = qc.discover_drafts(drafts_dir=drafts_dir)
        if missing:
            print(
                json.dumps(
                    {
                        "error": "Missing required drafts for --run-all",
                        "missing": [str(p) for p in missing],
                        "drafts_dir": str(drafts_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(1)
    else:
        drafts = [_resolve_single_test(args.test, drafts_dir=drafts_dir)]

    if not drafts:
        print(json.dumps({"error": "No draft documents found for processing"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    started = time.time()
    result = qc.run_for_drafts(drafts)
    elapsed_ms = int((time.time() - started) * 1000)
    runs = result["report"].get("runs", [])
    any_pipeline_failed = any(r.get("pipeline_status") != "ok" for r in runs)
    any_gate_failed = any(not r.get("gate_decision", {}).get("ready", False) for r in runs)
    final_status = "failed" if (any_pipeline_failed or any_gate_failed) else "ok"
    summary = {
        "status": final_status,
        "elapsed_ms": elapsed_ms,
        "drafts_processed": len(drafts),
        "output_files": result["output_files"],
        "aggregate_checks": result["report"]["aggregate_checks"],
        "aggregate_metrics": result["report"].get("aggregate_metrics", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if final_status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
