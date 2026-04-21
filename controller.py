"""Central pipeline orchestrator for STOFabric (Stage 6)."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from content_transformer import (
    DEFAULT_MODEL,
    MODEL_FLASH,
    MODEL_PRO,
    ContentTransformer,
)
from formatter import Formatter
from parser import parse_document
from semantic_mapper import SemanticMapper

try:
    from rich.console import Console
except Exception:  # pragma: no cover
    Console = None  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

PipelineMode = Literal["full", "up-to-mapper", "up-to-transformer"]


@dataclass(slots=True)
class StageResult:
    """One stage execution record."""

    name: str
    status: Literal["ok", "failed", "skipped"]
    elapsed_ms: int
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    """Runtime context for pipeline execution."""

    input_docx: Path
    output_docx: Path
    rules_path: Path
    schema_path: Path
    mode: PipelineMode
    model_key: Literal["8b", "70b", "405b"]
    save_intermediate: bool
    verbose: bool
    dry_run: bool
    output_dir: Path
    mapper_json_path: Path
    transformed_json_path: Path


class STOFabricPipeline:
    """Runs parser -> mapper -> transformer -> formatter in one pipeline."""

    def __init__(self, context: PipelineContext) -> None:
        self.ctx = context
        self.stage_results: list[StageResult] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.llm_requests: int = 0
        self.parsed_doc: dict[str, Any] | None = None
        self.mapper_result: dict[str, Any] | None = None
        self.transformed_result: dict[str, Any] | None = None
        self.last_successful_stage: str | None = None
        self.console = Console() if Console else None

    def _log(self, message: str) -> None:
        if not self.ctx.verbose:
            return
        if self.console:
            self.console.print(f"[cyan]{message}[/cyan]")
        else:
            print(f"[INFO] {message}")

    def _stage_line(self, stage: StageResult) -> str:
        symbol = {"ok": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(stage.status, "INFO")
        return f"{symbol} {stage.name} ({stage.elapsed_ms} ms)"

    def _print_stage_result(self, stage: StageResult) -> None:
        if self.console:
            color = {"ok": "green", "failed": "red", "skipped": "yellow"}.get(stage.status, "white")
            self.console.print(f"[{color}]{self._stage_line(stage)}[/{color}]")
            if self.ctx.verbose and stage.summary:
                self.console.print(json.dumps(stage.summary, ensure_ascii=False, indent=2))
        else:
            print(self._stage_line(stage))
            if self.ctx.verbose and stage.summary:
                print(json.dumps(stage.summary, ensure_ascii=False, indent=2))

    def _record_stage(
        self,
        name: str,
        status: Literal["ok", "failed", "skipped"],
        started_at: float,
        summary: dict[str, Any] | None = None,
    ) -> None:
        elapsed_ms = int((time.time() - started_at) * 1000)
        self.stage_results.append(
            StageResult(
                name=name,
                status=status,
                elapsed_ms=elapsed_ms,
                summary=summary or {},
            )
        )
        if status == "ok":
            self.last_successful_stage = name
        self._print_stage_result(self.stage_results[-1])

    def _require_file(self, path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    def _resolve_model(self) -> str:
        if self.ctx.model_key == "8b":
            return MODEL_FLASH
        if self.ctx.model_key == "70b":
            return MODEL_PRO
        return DEFAULT_MODEL

    def run_parser(self) -> None:
        started = time.time()
        self._log("Running parser stage...")
        self.parsed_doc = parse_document(self.ctx.input_docx)
        summary = {
            "blocks_count": self.parsed_doc.get("metadata", {}).get("blocks_count", 0),
            "page_index_mode": self.parsed_doc.get("metadata", {}).get("page_index_mode"),
        }
        self._record_stage("parser", "ok", started, summary)

    def run_mapper(self) -> None:
        started = time.time()
        self._log("Running semantic mapper stage...")
        if self.parsed_doc is None:
            raise RuntimeError("Parser output is not available")
        rules = SemanticMapper.load_mapping_rules(self.ctx.rules_path)
        schema = SemanticMapper.load_schema(self.ctx.schema_path)
        mapper = SemanticMapper(rules)
        self.mapper_result = mapper.map_blocks(self.parsed_doc, schema=schema)
        mapper_warnings = self.mapper_result.get("warnings", [])
        mapper_errors = self.mapper_result.get("errors", [])
        self.warnings.extend(mapper_warnings)
        self.errors.extend(mapper_errors)
        if self.ctx.save_intermediate:
            self.ctx.mapper_json_path.write_text(
                json.dumps(self.mapper_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary = {
            "mapping_trace": len(self.mapper_result.get("mapping_trace", [])),
            "warnings": len(mapper_warnings),
            "errors": len(mapper_errors),
            "saved_json": str(self.ctx.mapper_json_path) if self.ctx.save_intermediate else None,
        }
        self._record_stage("semantic_mapper", "ok", started, summary)

    def run_transformer(self) -> None:
        started = time.time()
        self._log("Running content transformer stage...")
        if self.mapper_result is None:
            raise RuntimeError("Mapper output is not available")
        api_key, using_legacy_key = _resolve_llm_api_key()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY (or legacy NVIDIA_API_KEY) is required for transformer stage. "
                "Set OPENROUTER_API_KEY and retry."
            )
        if using_legacy_key:
            self.warnings.append("Using legacy NVIDIA_API_KEY fallback; prefer OPENROUTER_API_KEY.")

        rules = ContentTransformer.load_yaml(self.ctx.rules_path)
        schema = ContentTransformer.load_schema(self.ctx.schema_path)
        transformer = ContentTransformer(
            rules=rules,
            schema=schema,
            api_key=api_key,
            model=self._resolve_model(),
            api_url=os.getenv("OPENROUTER_API_URL", os.getenv("NVIDIA_API_URL", "https://openrouter.ai/api/v1/chat/completions")),
            timeout_seconds=int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "90"))),
            retries=int(os.getenv("OPENROUTER_RETRIES", os.getenv("NVIDIA_RETRIES", "4"))),
            retry_backoff_base_seconds=float(os.getenv("OPENROUTER_RETRY_BACKOFF_BASE_SECONDS", os.getenv("NVIDIA_RETRY_BACKOFF_BASE_SECONDS", "2.0"))),
            max_input_chars_per_request=int(os.getenv("OPENROUTER_MAX_INPUT_CHARS_PER_REQUEST", os.getenv("NVIDIA_MAX_INPUT_CHARS_PER_REQUEST", "2200"))),
            chunk_overlap_chars=int(os.getenv("OPENROUTER_CHUNK_OVERLAP_CHARS", os.getenv("NVIDIA_CHUNK_OVERLAP_CHARS", "180"))),
            fail_on_rewrite_error=bool(_env_bool("OPENROUTER_FAIL_ON_REWRITE_ERROR") or _env_bool("NVIDIA_FAIL_ON_REWRITE_ERROR") or False),
            rewrite_strategy=os.getenv("OPENROUTER_REWRITE_STRATEGY", os.getenv("NVIDIA_REWRITE_STRATEGY", "single_pass")),
            fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL", os.getenv("NVIDIA_FALLBACK_MODEL", MODEL_FLASH)),
            fallback_after_timeouts=int(os.getenv("OPENROUTER_FALLBACK_AFTER_TIMEOUTS", os.getenv("NVIDIA_FALLBACK_AFTER_TIMEOUTS", "2"))),
        )
        self.transformed_result = transformer.transform(self.mapper_result)
        report = self.transformed_result.get("transform_report", {})
        tr_warnings = report.get("warnings", [])
        tr_errors = report.get("errors", [])
        self.warnings.extend(tr_warnings)
        self.errors.extend(tr_errors)
        self.llm_requests += int(report.get("model_metadata", {}).get("requests", 0))
        if self.ctx.save_intermediate:
            self.ctx.transformed_json_path.write_text(
                json.dumps(self.transformed_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary = {
            "model": report.get("model_metadata", {}).get("model"),
            "llm_requests": report.get("model_metadata", {}).get("requests", 0),
            "llm_attempts_total": report.get("model_metadata", {}).get("attempts_total", 0),
            "llm_timeouts_total": report.get("model_metadata", {}).get("timeouts_total", 0),
            "llm_http_errors_total": report.get("model_metadata", {}).get("http_errors_total", 0),
            "llm_fallback_activations": report.get("model_metadata", {}).get("fallback_activations", 0),
            "rewrite_strategy": report.get("model_metadata", {}).get("rewrite_strategy"),
            "warnings": len(tr_warnings),
            "errors": len(tr_errors),
            "saved_json": str(self.ctx.transformed_json_path) if self.ctx.save_intermediate else None,
        }
        self._record_stage("content_transformer", "ok", started, summary)

    def run_formatter(self) -> None:
        started = time.time()
        self._log("Running formatter stage...")
        if self.transformed_result is None:
            raise RuntimeError("Transformer output is not available")
        formatter = Formatter(self.transformed_result)
        result = formatter.build(self.ctx.output_docx)
        self.warnings.extend(result.get("warnings", []))
        summary = {
            "output_docx": str(self.ctx.output_docx),
            "warnings": len(result.get("warnings", [])),
        }
        self._record_stage("formatter", "ok", started, summary)

    def run(self) -> dict[str, Any]:
        total_started = time.time()
        try:
            self._require_file(self.ctx.input_docx, "Input DOCX")
            self._require_file(self.ctx.rules_path, "Rules YAML")
            self._require_file(self.ctx.schema_path, "Schema JSON")

            self.run_parser()
            self.run_mapper()

            if self.ctx.dry_run:
                self._record_stage("content_transformer", "skipped", time.time(), {"reason": "dry-run"})
                self._record_stage("formatter", "skipped", time.time(), {"reason": "dry-run"})
                return self._build_summary(total_started)

            if self.ctx.mode == "up-to-mapper":
                self._record_stage("content_transformer", "skipped", time.time(), {"reason": "mode=up-to-mapper"})
                self._record_stage("formatter", "skipped", time.time(), {"reason": "mode=up-to-mapper"})
                return self._build_summary(total_started)

            self.run_transformer()

            if self.ctx.mode == "up-to-transformer":
                self._record_stage("formatter", "skipped", time.time(), {"reason": "mode=up-to-transformer"})
                return self._build_summary(total_started)

            self.run_formatter()
            return self._build_summary(total_started)
        except Exception as exc:
            self.errors.append(str(exc))
            failed_stage = self._infer_failed_stage()
            hint = self._build_failure_hint(failed_stage, str(exc))
            self.stage_results.append(
                StageResult(
                    name=failed_stage,
                    status="failed",
                    elapsed_ms=0,
                    summary={"message": str(exc), "hint": hint},
                )
            )
            self._print_stage_result(self.stage_results[-1])
            return self._build_summary(
                total_started,
                failed=True,
                failure_message=str(exc),
                failure_stage=failed_stage,
                failure_hint=hint,
            )

    def _infer_failed_stage(self) -> str:
        done = {s.name for s in self.stage_results}
        if "parser" not in done:
            return "parser"
        if "semantic_mapper" not in done:
            return "semantic_mapper"
        if self.ctx.mode != "up-to-mapper" and "content_transformer" not in done:
            return "content_transformer"
        if self.ctx.mode == "full" and "formatter" not in done:
            return "formatter"
        return "pipeline"

    def _build_failure_hint(self, stage: str, error_message: str) -> str:
        lowered = error_message.lower()
        if "nvidia_api_key" in lowered or "api key" in lowered:
            return "Set OPENROUTER_API_KEY (or legacy NVIDIA_API_KEY) and retry (or run --dry-run/--mode up-to-mapper)."
        if "not found" in lowered:
            return "Verify file paths for --input-docx, --rules, --schema and --output-docx."
        if stage == "content_transformer":
            return "Re-run with --mode up-to-mapper to verify pre-transform artifacts, then retry transformer."
        return "Re-run with --verbose to inspect per-stage summaries and fix the failed stage."

    def _build_summary(
        self,
        total_started: float,
        failed: bool = False,
        failure_message: str | None = None,
        failure_stage: str | None = None,
        failure_hint: str | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed" if failed else "ok",
            "mode": self.ctx.mode,
            "dry_run": self.ctx.dry_run,
            "input_docx": str(self.ctx.input_docx),
            "output_docx": str(self.ctx.output_docx),
            "elapsed_ms": int((time.time() - total_started) * 1000),
            "llm_requests": self.llm_requests,
            "nvidia_requests": self.llm_requests,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "last_successful_stage": self.last_successful_stage,
            "warnings": self.warnings,
            "errors": self.errors,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "elapsed_ms": s.elapsed_ms,
                    "summary": s.summary,
                }
                for s in self.stage_results
            ],
            "artifacts": {
                "mapper_json": str(self.ctx.mapper_json_path) if self.ctx.save_intermediate else None,
                "transformed_json": str(self.ctx.transformed_json_path) if self.ctx.save_intermediate else None,
                "output_docx": str(self.ctx.output_docx) if self.ctx.mode == "full" else None,
            },
            "failure": {
                "stage": failure_stage,
                "message": failure_message,
                "hint": failure_hint,
            }
            if failed
            else None,
        }


def _build_cli() -> argparse.ArgumentParser:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="STOFabric central orchestrator")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    parser.add_argument("--input-docx", required=False, type=Path, default=None, help="Path to draft DOCX")
    parser.add_argument("--output-docx", required=False, type=Path, default=None, help="Path to output STO DOCX")
    parser.add_argument("--rules", default=None, type=Path, help="Path to mapping rules YAML")
    parser.add_argument("--schema", default=None, type=Path, help="Path to STO schema JSON")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["full", "up-to-mapper", "up-to-transformer"],
        help="Pipeline execution mode",
    )
    parser.add_argument("--model", default=None, choices=["8b", "70b", "405b"], help="LLM model key (8b=flash,70b=pro)")
    parser.add_argument("--save-intermediate", dest="save_intermediate", action="store_true", help="Save mapper/transformed JSON artifacts")
    parser.add_argument("--no-save-intermediate", dest="save_intermediate", action="store_false", help="Disable mapper/transformed JSON artifacts")
    parser.set_defaults(save_intermediate=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Run only parser + mapper; skip transformer and formatter")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Disable dry-run mode")
    parser.set_defaults(dry_run=None)
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Verbose stage logs")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose logs")
    parser.set_defaults(verbose=None)
    return parser


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping/object")
    return data


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_llm_api_key() -> tuple[str, bool]:
    """Resolve API key with OpenRouter priority and legacy fallback."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        return openrouter_key, False
    legacy_key = os.getenv("NVIDIA_API_KEY", "")
    if legacy_key:
        return legacy_key, True
    return "", False


def _build_context(args: argparse.Namespace) -> PipelineContext:
    config = _load_config(args.config)
    input_docx_raw = _coalesce(args.input_docx, os.getenv("STO_INPUT_DOCX"), config.get("input_docx"))
    output_docx_raw = _coalesce(args.output_docx, os.getenv("STO_OUTPUT_DOCX"), config.get("output_docx"))
    if input_docx_raw is None:
        raise ValueError("Input DOCX is required. Use --input-docx or STO_INPUT_DOCX.")
    if output_docx_raw is None:
        raise ValueError("Output DOCX is required. Use --output-docx or STO_OUTPUT_DOCX.")
    input_docx = Path(input_docx_raw)
    output_docx = Path(output_docx_raw)

    rules = Path(_coalesce(args.rules, os.getenv("STO_RULES_PATH"), config.get("rules"), "mapping-rules.yaml"))
    schema = Path(_coalesce(args.schema, os.getenv("STO_SCHEMA_PATH"), config.get("schema"), "sto-model.schema.json"))
    mode = _coalesce(args.mode, os.getenv("STO_MODE"), config.get("mode"), "full")
    model = _coalesce(args.model, os.getenv("STO_MODEL_KEY"), config.get("model"), "70b")
    save_intermediate = bool(
        _coalesce(args.save_intermediate, _env_bool("STO_SAVE_INTERMEDIATE"), config.get("save_intermediate"), False)
    )
    verbose = bool(_coalesce(args.verbose, _env_bool("STO_VERBOSE"), config.get("verbose"), False))
    dry_run = bool(_coalesce(args.dry_run, _env_bool("STO_DRY_RUN"), config.get("dry_run"), False))

    output_dir = Path("_tmp_extract")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapper_json = output_dir / "mapper.json"
    transformed_json = output_dir / "transformed.json"
    return PipelineContext(
        input_docx=input_docx,
        output_docx=output_docx,
        rules_path=rules,
        schema_path=schema,
        mode=mode,
        model_key=model,
        save_intermediate=save_intermediate,
        verbose=verbose,
        dry_run=dry_run,
        output_dir=output_dir,
        mapper_json_path=mapper_json,
        transformed_json_path=transformed_json,
    )


def main() -> None:
    args = _build_cli().parse_args()
    ctx = _build_context(args)
    pipeline = STOFabricPipeline(ctx)
    result = pipeline.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
