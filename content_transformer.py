"""Content Transformer agent for STOFabric stage 5.

Transforms mapper JSON into official business style text while preserving structure.
Uses OpenRouter API (Gemini family) for text rewriting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from yaml import YAMLError

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


DEFAULT_MODEL = "google/gemini-2.5-pro"
MODEL_FLASH = "google/gemini-2.5-flash"
MODEL_PRO = DEFAULT_MODEL
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ContentTransformer:
    """Transforms sto_document_json using OpenRouter LLM and schema-aware normalization."""

    def __init__(
        self,
        rules: dict[str, Any],
        schema: dict[str, Any] | None,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_OPENROUTER_URL,
        timeout_seconds: int = 90,
        retries: int = 4,
        retry_backoff_base_seconds: float = 2.0,
        max_input_chars_per_request: int = 2200,
        chunk_overlap_chars: int = 180,
        fail_on_rewrite_error: bool = False,
        rewrite_strategy: str = "single_pass",
        fallback_model: str | None = MODEL_FLASH,
        fallback_after_timeouts: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required for content transformation")
        self.rules = rules.get("mapping_rules", rules)
        self.schema = schema
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.max_input_chars_per_request = max_input_chars_per_request
        self.chunk_overlap_chars = chunk_overlap_chars
        self.fail_on_rewrite_error = fail_on_rewrite_error
        self.rewrite_strategy = rewrite_strategy
        self.fallback_model = fallback_model
        self.fallback_after_timeouts = fallback_after_timeouts
        self.report: dict[str, Any] = {
            "warnings": [],
            "errors": [],
            "applied_defaults": [],
            "model_metadata": {
                "provider": "openrouter",
                "model": model,
                "api_url": api_url,
                "rewrite_strategy": rewrite_strategy,
                "timeout_seconds": timeout_seconds,
                "retries": retries,
                "requests": 0,
                "attempts_total": 0,
                "timeouts_total": 0,
                "http_errors_total": 0,
                "requests_succeeded": 0,
                "fallback_model": fallback_model,
                "fallback_activations": 0,
            },
            "rewrite_trace": [],
        }

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        try:
            return yaml.safe_load(text)
        except YAMLError:
            # Keep transformer resilient to known indentation drift in rules file.
            fixed = text.replace("\n    conflict_resolution:\n", "\n  conflict_resolution:\n")
            parsed = yaml.safe_load(fixed)
            if isinstance(parsed, dict):
                parsed.setdefault("mapping_rules", {}).setdefault("_load_diagnostics", []).append(
                    "rules_loaded_with_fallback_indentation_fix"
                )
            return parsed

    @staticmethod
    def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return [text]
        if len(cleaned) <= max_chars:
            return [cleaned]

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
                continue
            # Fallback when one paragraph exceeds limit.
            start = 0
            step = max(1, max_chars - overlap_chars)
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start += step
            current = ""
        if current:
            chunks.append(current)
        return chunks or [cleaned]

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def load_schema(path: str | Path | None) -> dict[str, Any] | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def ensure_required_structure(self, sto_document: dict[str, Any]) -> dict[str, Any]:
        """Ensure schema-required structure and set safe defaults."""
        doc = sto_document
        doc.setdefault(
            "meta",
            {
                "sto_number": "СТО 31025229-000-2024",
                "title": "Черновик СТО (автогенерация)",
                "version": "draft",
                "approval_date": "1970-01-01",
                "status": "draft",
            },
        )
        doc["meta"].setdefault("extra_attributes", {})
        doc["meta"]["extra_attributes"].setdefault("service_sheets", {})
        doc["meta"]["extra_attributes"].setdefault("mapping_diagnostics", {})

        doc.setdefault(
            "content",
            {
                "area": "",
                "normative_references": [],
                "terms": {"intro": "", "entries": [], "abbreviations": []},
                "main": [],
            },
        )
        doc["content"].setdefault("area", "")
        doc["content"].setdefault("normative_references", [])
        doc["content"].setdefault("terms", {"intro": "", "entries": [], "abbreviations": []})
        doc["content"]["terms"].setdefault("entries", [])
        doc["content"]["terms"].setdefault("abbreviations", [])
        doc["content"]["terms"].setdefault("intro", "")
        doc["content"].setdefault("main", [])
        doc["content"].setdefault("document_flow_diagrams", [])
        doc["content"].setdefault("extra_attributes", {})

        doc.setdefault("reporting_documents", [])
        doc.setdefault("responsibility", {"title": "Ответственность", "entries": []})
        doc["responsibility"]["title"] = "Ответственность"
        doc["responsibility"].setdefault("entries", [])
        doc.setdefault("appendices", [])
        return doc

    def normalize_service_sheets(self, result: dict[str, Any], sto_document: dict[str, Any]) -> None:
        """Move service sheets to canonical path."""
        canonical = sto_document["meta"]["extra_attributes"].setdefault("service_sheets", {})
        for key in ("change_log", "acquaintance", "approval"):
            canonical.setdefault(key, [])
        external = result.get("service_sheet_extract") or {}
        for key in ("change_log", "acquaintance", "approval"):
            if external.get(key):
                canonical[key] = external[key]
                self.report["applied_defaults"].append(f"service_sheets.{key} synchronized from service_sheet_extract")

    def fill_missing_required_sections(self, sto_document: dict[str, Any]) -> None:
        """Fill mandatory sections with controlled placeholders."""
        content = sto_document["content"]
        if not content["area"]:
            content["area"] = (
                "Настоящий стандарт устанавливает единый порядок выполнения процессов "
                "и обязателен для применения в пределах установленной области ответственности."
            )
            self.report["applied_defaults"].append("content.area placeholder added")
        if not content["normative_references"]:
            content["normative_references"].append(
                {"reference_id": "СТО 31025229-001-2024", "title": "Стандарты организации. Общие требования"}
            )
            self.report["applied_defaults"].append("content.normative_references placeholder added")
        if not content["terms"]["entries"] and not content["terms"]["abbreviations"]:
            content["terms"]["intro"] = content["terms"]["intro"] or (
                "В настоящем стандарте применяются термины и сокращения в соответствии с действующими нормативными документами."
            )
            self.report["applied_defaults"].append("content.terms intro placeholder added")
        if not content["main"]:
            content["main"].append(
                {
                    "section_number": "4",
                    "title": "Основные требования",
                    "content": "Требования основной части подлежат уточнению на основании исходного черновика.",
                    "subsections": [],
                    "tables": [],
                    "images": [],
                    "bullet_points": [],
                }
            )
            self.report["applied_defaults"].append("content.main placeholder section added")
        if not sto_document["responsibility"]["entries"]:
            sto_document["responsibility"]["entries"].append(
                {"role": "Руководитель процесса", "responsibilities": ["Обеспечивает выполнение требований настоящего стандарта."]}
            )
            self.report["applied_defaults"].append("responsibility placeholder added")

    def _call_llm(self, text: str, section_hint: str, chunk_idx: int | None, model: str) -> str:
        """Call LLM endpoint for one chunk with retry/backoff."""
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты редактор корпоративных нормативных документов. "
                        "Переписывай текст строго в официально-деловом стиле СТО 31025229-001-2024 "
                        "и формулировках, типичных для СТО 002, 003, 004, 005, 007, 010, 012. "
                        "Используй безличные и нормативные конструкции: 'настоящий стандарт устанавливает', "
                        "'должен', 'подлежит', 'обеспечивает'. "
                        "Запрещены разговорные обороты, эмоциональные или оценочные формулировки. "
                        "Обязательно сохраняй 100% смысла исходника: точные термины, обозначения, "
                        "номера пунктов, коды документов, роли, сроки, условия, ссылки и перечисления. "
                        "Не сокращай фактическое содержание и не добавляй вымышленные данные."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Раздел: {section_hint}\n"
                        "Перепиши текст ниже в стиле СТО 31025229-001-2024.\n"
                        "Используй официальный деловой язык, без разговорных оборотов.\n"
                        "Не изменяй факты, числа, идентификаторы, номера пунктов, термины и ссылки.\n"
                        "Сохрани исходную логическую структуру и последовательность требований.\n"
                        "Верни только отредактированный текст без комментариев.\n\n"
                        f"{text}"
                    ),
                },
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = Request(self.api_url, data=body, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.report["model_metadata"]["attempts_total"] += 1
            try:
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.report["model_metadata"]["requests"] += 1
                self.report["model_metadata"]["requests_succeeded"] += 1
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("LLM response has no choices")
                message = choices[0].get("message", {})
                rewritten = (message.get("content") or "").strip()
                if not rewritten:
                    raise RuntimeError("LLM response content is empty")
                self.report["rewrite_trace"].append(
                    {
                        "section_hint": section_hint,
                        "chunk_idx": chunk_idx,
                        "model": model,
                        "status": "ok",
                    }
                )
                return rewritten
            except TimeoutError as exc:
                last_error = exc
                self.report["model_metadata"]["timeouts_total"] += 1
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff_base_seconds * (2**attempt))
            except HTTPError as exc:
                last_error = exc
                self.report["model_metadata"]["http_errors_total"] += 1
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff_base_seconds * (2**attempt))
            except (URLError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff_base_seconds * (2**attempt))
        raise RuntimeError(
            f"LLM API call failed section={section_hint} chunk={chunk_idx} model={model} after retries: {last_error}"
        )

    def _rewrite_text_with_llm(self, text: str, section_hint: str) -> str:
        """Rewrite text with chunking and optional fallback model."""
        if not text.strip():
            return text
        chunks = self._chunk_text(text, self.max_input_chars_per_request, self.chunk_overlap_chars)
        out_chunks: list[str] = []
        for idx, chunk in enumerate(chunks):
            self.report["rewrite_trace"].append(
                {
                    "section_hint": section_hint,
                    "chunk_idx": idx,
                    "model": self.model,
                    "status": "start",
                }
            )
            try:
                out_chunks.append(self._call_llm(chunk, section_hint, idx, self.model))
            except RuntimeError as exc:
                use_fallback = (
                    self.fallback_model
                    and self.fallback_model != self.model
                    and self.report["model_metadata"]["timeouts_total"] >= self.fallback_after_timeouts
                )
                if use_fallback:
                    self.report["model_metadata"]["fallback_activations"] += 1
                    try:
                        out_chunks.append(self._call_llm(chunk, section_hint, idx, self.fallback_model))
                        continue
                    except RuntimeError as fallback_exc:
                        exc = fallback_exc
                self.report["rewrite_trace"].append(
                    {
                        "section_hint": section_hint,
                        "chunk_idx": idx,
                        "model": self.model,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if self.fail_on_rewrite_error:
                    raise
                self.report["warnings"].append(
                    f"rewrite_failed section={section_hint} chunk={idx}; original text preserved; reason={exc}"
                )
                out_chunks.append(chunk)
        return "\n".join(out_chunks).strip()

    def rewrite_official_style_with_llm(self, sto_document: dict[str, Any], critical_only: bool = False) -> None:
        """Apply LLM rewriting to core textual fields while preserving structure."""
        content = sto_document["content"]
        content["area"] = self._rewrite_text_with_llm(content["area"], "content.area")

        if not critical_only:
            for idx, ref in enumerate(content["normative_references"]):
                title = ref.get("title", "")
                if title:
                    ref["title"] = self._rewrite_text_with_llm(title, f"content.normative_references[{idx}].title")

        terms = content["terms"]
        if terms.get("intro"):
            terms["intro"] = self._rewrite_text_with_llm(terms["intro"], "content.terms.intro")
        if not critical_only:
            for idx, entry in enumerate(terms.get("entries", [])):
                if entry.get("definition"):
                    entry["definition"] = self._rewrite_text_with_llm(
                        entry["definition"], f"content.terms.entries[{idx}].definition"
                    )
        for idx, section in enumerate(content.get("main", [])):
            if section.get("content"):
                section["content"] = self._rewrite_text_with_llm(section["content"], f"content.main[{idx}].content")
            for sub_idx, sub in enumerate(section.get("subsections", [])):
                if sub.get("content"):
                    sub["content"] = self._rewrite_text_with_llm(
                        sub["content"], f"content.main[{idx}].subsections[{sub_idx}].content"
                    )

        for idx, entry in enumerate(sto_document["responsibility"].get("entries", [])):
            entry["responsibilities"] = [
                self._rewrite_text_with_llm(text, f"responsibility.entries[{idx}]") for text in entry.get("responsibilities", [])
            ]

        if not critical_only:
            for idx, row in enumerate(sto_document.get("reporting_documents", [])):
                if row.get("notes"):
                    row["notes"] = self._rewrite_text_with_llm(row["notes"], f"reporting_documents[{idx}].notes")

            for idx, appendix in enumerate(sto_document.get("appendices", [])):
                if appendix.get("content_text"):
                    appendix["content_text"] = self._rewrite_text_with_llm(
                        appendix["content_text"], f"appendices[{idx}].content_text"
                    )

    def validate_schema(self, sto_document: dict[str, Any]) -> None:
        """Validate output JSON against schema if validator is available."""
        if self.schema is None:
            self.report["warnings"].append("Schema file is unavailable; schema validation skipped")
            return
        if jsonschema is None:
            self.report["warnings"].append("jsonschema is not installed; strict schema validation skipped")
            return
        try:
            jsonschema.validate(sto_document, self.schema)
        except Exception as exc:
            self.report["errors"].append(f"Schema validation failed: {exc}")

    def transform(self, mapper_result: dict[str, Any]) -> dict[str, Any]:
        """Run full transformation pipeline and return transformed JSON report."""
        source = mapper_result.get("sto_document_json", mapper_result)
        sto_document = deepcopy(source)
        sto_document = self.ensure_required_structure(sto_document)
        self.normalize_service_sheets(mapper_result, sto_document)
        self.fill_missing_required_sections(sto_document)
        if self.rewrite_strategy == "two_pass_8b_70b":
            original_model = self.model
            self.model = MODEL_FLASH
            self.rewrite_official_style_with_llm(sto_document, critical_only=False)
            self.model = original_model
            self.rewrite_official_style_with_llm(sto_document, critical_only=True)
        else:
            self.rewrite_official_style_with_llm(sto_document)
        self.validate_schema(sto_document)
        return {"sto_document_json": sto_document, "transform_report": self.report}


def _build_cli() -> argparse.ArgumentParser:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Transform mapped STO JSON with OpenRouter API")
    parser.add_argument("--input-json", required=True, type=Path, help="Mapper output JSON path")
    parser.add_argument("--rules", required=True, type=Path, help="mapping-rules.yaml path")
    parser.add_argument("--schema", default=Path("sto-model.schema.json"), type=Path, help="Schema JSON path")
    parser.add_argument("--output-json", required=True, type=Path, help="Output transformed JSON path")
    parser.add_argument(
        "--api-key-env",
        default=os.getenv("OPENROUTER_API_KEY_ENV", os.getenv("NVIDIA_API_KEY_ENV", "OPENROUTER_API_KEY")),
        help="Environment variable containing LLM API key",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)),
        help="LLM model id",
    )
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "90"))))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPENROUTER_RETRIES", os.getenv("NVIDIA_RETRIES", "4"))))
    parser.add_argument("--retry-backoff-base-seconds", type=float, default=float(os.getenv("OPENROUTER_RETRY_BACKOFF_BASE_SECONDS", os.getenv("NVIDIA_RETRY_BACKOFF_BASE_SECONDS", "2.0"))))
    parser.add_argument("--max-input-chars-per-request", type=int, default=int(os.getenv("OPENROUTER_MAX_INPUT_CHARS_PER_REQUEST", os.getenv("NVIDIA_MAX_INPUT_CHARS_PER_REQUEST", "2200"))))
    parser.add_argument("--chunk-overlap-chars", type=int, default=int(os.getenv("OPENROUTER_CHUNK_OVERLAP_CHARS", os.getenv("NVIDIA_CHUNK_OVERLAP_CHARS", "180"))))
    parser.add_argument("--rewrite-strategy", choices=["single_pass", "two_pass_8b_70b"], default=os.getenv("OPENROUTER_REWRITE_STRATEGY", os.getenv("NVIDIA_REWRITE_STRATEGY", "single_pass")))
    parser.add_argument("--fail-on-rewrite-error", dest="fail_on_rewrite_error", action="store_true")
    parser.add_argument("--no-fail-on-rewrite-error", dest="fail_on_rewrite_error", action="store_false")
    parser.set_defaults(fail_on_rewrite_error=os.getenv("OPENROUTER_FAIL_ON_REWRITE_ERROR", os.getenv("NVIDIA_FAIL_ON_REWRITE_ERROR", "")).strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--fallback-model", default=os.getenv("OPENROUTER_FALLBACK_MODEL", os.getenv("NVIDIA_FALLBACK_MODEL", MODEL_FLASH)))
    parser.add_argument("--fallback-after-timeouts", type=int, default=int(os.getenv("OPENROUTER_FALLBACK_AFTER_TIMEOUTS", os.getenv("NVIDIA_FALLBACK_AFTER_TIMEOUTS", "2"))))
    parser.add_argument(
        "--api-url",
        default=os.getenv("OPENROUTER_API_URL", os.getenv("NVIDIA_API_URL", DEFAULT_OPENROUTER_URL)),
        help="OpenRouter API URL",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        legacy_key = os.getenv("NVIDIA_API_KEY", "")
        api_key = legacy_key
    if not api_key:
        print(
            json.dumps(
                {"error": f"Environment variable {args.api_key_env} is required for real OpenRouter API mode"},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1)

    try:
        mapper_result = ContentTransformer.load_json(args.input_json)
        rules = ContentTransformer.load_yaml(args.rules)
        schema = ContentTransformer.load_schema(args.schema)
        transformer = ContentTransformer(
            rules=rules,
            schema=schema,
            api_key=api_key,
            model=args.model,
            api_url=args.api_url,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_backoff_base_seconds=args.retry_backoff_base_seconds,
            max_input_chars_per_request=args.max_input_chars_per_request,
            chunk_overlap_chars=args.chunk_overlap_chars,
            fail_on_rewrite_error=args.fail_on_rewrite_error,
            rewrite_strategy=args.rewrite_strategy,
            fallback_model=args.fallback_model,
            fallback_after_timeouts=args.fallback_after_timeouts,
        )
        transformed = transformer.transform(mapper_result)
        args.output_json.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "output_json": str(args.output_json),
            "errors": len(transformed["transform_report"].get("errors", [])),
            "warnings": len(transformed["transform_report"].get("warnings", [])),
            "model": args.model,
            "requests": transformed["transform_report"].get("model_metadata", {}).get("requests", 0),
        }
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
