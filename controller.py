"""Central pipeline orchestrator for STOFabric (Stage 6)."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from content_transformer import (
    DEFAULT_MODEL,
    MODEL_405B,
    ContentTransformer,
)
from formatter import Formatter
from parser import parse_document
from semantic_mapper import SemanticMapper

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
    model_key: Literal["70b", "405b"]
    save_intermediate: bool
    verbose: bool
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
        self.nvidia_requests: int = 0
        self.parsed_doc: dict[str, Any] | None = None
        self.mapper_result: dict[str, Any] | None = None
        self.transformed_result: dict[str, Any] | None = None

    def _log(self, message: str) -> None:
        if self.ctx.verbose:
            print(message)

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

    def _require_file(self, path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    def _resolve_model(self) -> str:
        return DEFAULT_MODEL if self.ctx.model_key == "70b" else MODEL_405B

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
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required for transformer stage. "
                "Set environment variable NVIDIA_API_KEY and retry."
            )

        rules = ContentTransformer.load_yaml(self.ctx.rules_path)
        schema = ContentTransformer.load_schema(self.ctx.schema_path)
        transformer = ContentTransformer(
            rules=rules,
            schema=schema,
            api_key=api_key,
            model=self._resolve_model(),
        )
        self.transformed_result = transformer.transform(self.mapper_result)
        report = self.transformed_result.get("transform_report", {})
        tr_warnings = report.get("warnings", [])
        tr_errors = report.get("errors", [])
        self.warnings.extend(tr_warnings)
        self.errors.extend(tr_errors)
        self.nvidia_requests += int(report.get("model_metadata", {}).get("requests", 0))
        if self.ctx.save_intermediate:
            self.ctx.transformed_json_path.write_text(
                json.dumps(self.transformed_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary = {
            "model": report.get("model_metadata", {}).get("model"),
            "nvidia_requests": report.get("model_metadata", {}).get("requests", 0),
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
            self.stage_results.append(
                StageResult(
                    name=failed_stage,
                    status="failed",
                    elapsed_ms=0,
                    summary={"message": str(exc)},
                )
            )
            return self._build_summary(total_started, failed=True, failure_message=str(exc), failure_stage=failed_stage)

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

    def _build_summary(
        self,
        total_started: float,
        failed: bool = False,
        failure_message: str | None = None,
        failure_stage: str | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed" if failed else "ok",
            "mode": self.ctx.mode,
            "input_docx": str(self.ctx.input_docx),
            "output_docx": str(self.ctx.output_docx),
            "elapsed_ms": int((time.time() - total_started) * 1000),
            "nvidia_requests": self.nvidia_requests,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
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
            }
            if failed
            else None,
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STOFabric central orchestrator")
    parser.add_argument("--input-docx", required=True, type=Path, help="Path to draft DOCX")
    parser.add_argument("--output-docx", required=True, type=Path, help="Path to output STO DOCX")
    parser.add_argument("--rules", default=Path("mapping-rules.yaml"), type=Path, help="Path to mapping rules YAML")
    parser.add_argument("--schema", default=Path("sto-model.schema.json"), type=Path, help="Path to STO schema JSON")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "up-to-mapper", "up-to-transformer"],
        help="Pipeline execution mode",
    )
    parser.add_argument("--model", default="70b", choices=["70b", "405b"], help="NVIDIA model size key")
    parser.add_argument("--save-intermediate", action="store_true", help="Save mapper/transformed JSON artifacts")
    parser.add_argument("--verbose", action="store_true", help="Verbose stage logs")
    return parser


def _build_context(args: argparse.Namespace) -> PipelineContext:
    output_dir = args.output_docx.parent
    mapper_json = output_dir / "mapper.json"
    transformed_json = output_dir / "transformed.json"
    return PipelineContext(
        input_docx=args.input_docx,
        output_docx=args.output_docx,
        rules_path=args.rules,
        schema_path=args.schema,
        mode=args.mode,
        model_key=args.model,
        save_intermediate=args.save_intermediate,
        verbose=args.verbose,
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
