"""Content Transformer agent for STOFabric stage 5.

Transforms mapper JSON into official business style text while preserving structure.
Uses NVIDIA Build API (Llama 3.1) for text rewriting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
MODEL_405B = "meta/llama-3.1-405b-instruct"
DEFAULT_NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class ContentTransformer:
    """Transforms sto_document_json using NVIDIA LLM and schema-aware normalization."""

    def __init__(
        self,
        rules: dict[str, Any],
        schema: dict[str, Any] | None,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_NVIDIA_URL,
        timeout_seconds: int = 45,
        retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("NVIDIA API key is required for content transformation")
        self.rules = rules.get("mapping_rules", rules)
        self.schema = schema
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.report: dict[str, Any] = {
            "warnings": [],
            "errors": [],
            "applied_defaults": [],
            "model_metadata": {
                "provider": "nvidia_build_api",
                "model": model,
                "api_url": api_url,
                "requests": 0,
            },
        }

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

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

    def _rewrite_text_with_nvidia(self, text: str, section_hint: str) -> str:
        """Rewrite a text block to official STO style via NVIDIA Build API."""
        if not text.strip():
            return text
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты редактор стандартов организации. Перефразируй текст в официально-деловом стиле СТО. "
                        "Сохраняй 100% смысла, фактов, ссылок, ролей, сроков и структуры."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Раздел: {section_hint}\n"
                        "Перепиши текст ниже в стиле СТО 31025229-001-2024.\n"
                        "Не добавляй вымышленные данные. Верни только отредактированный текст.\n\n"
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
            try:
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.report["model_metadata"]["requests"] += 1
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("NVIDIA response has no choices")
                message = choices[0].get("message", {})
                rewritten = (message.get("content") or "").strip()
                if not rewritten:
                    raise RuntimeError("NVIDIA response content is empty")
                return rewritten
            except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"NVIDIA API call failed after retries: {last_error}")

    def rewrite_official_style_with_nvidia(self, sto_document: dict[str, Any]) -> None:
        """Apply LLM rewriting to core textual fields while preserving structure."""
        content = sto_document["content"]
        content["area"] = self._rewrite_text_with_nvidia(content["area"], "content.area")

        for idx, ref in enumerate(content["normative_references"]):
            title = ref.get("title", "")
            if title:
                ref["title"] = self._rewrite_text_with_nvidia(title, f"content.normative_references[{idx}].title")

        terms = content["terms"]
        if terms.get("intro"):
            terms["intro"] = self._rewrite_text_with_nvidia(terms["intro"], "content.terms.intro")
        for idx, entry in enumerate(terms.get("entries", [])):
            if entry.get("definition"):
                entry["definition"] = self._rewrite_text_with_nvidia(
                    entry["definition"], f"content.terms.entries[{idx}].definition"
                )
        for idx, section in enumerate(content.get("main", [])):
            if section.get("content"):
                section["content"] = self._rewrite_text_with_nvidia(section["content"], f"content.main[{idx}].content")
            for sub_idx, sub in enumerate(section.get("subsections", [])):
                if sub.get("content"):
                    sub["content"] = self._rewrite_text_with_nvidia(
                        sub["content"], f"content.main[{idx}].subsections[{sub_idx}].content"
                    )

        for idx, entry in enumerate(sto_document["responsibility"].get("entries", [])):
            entry["responsibilities"] = [
                self._rewrite_text_with_nvidia(text, f"responsibility.entries[{idx}]") for text in entry.get("responsibilities", [])
            ]

        for idx, row in enumerate(sto_document.get("reporting_documents", [])):
            if row.get("notes"):
                row["notes"] = self._rewrite_text_with_nvidia(row["notes"], f"reporting_documents[{idx}].notes")

        for idx, appendix in enumerate(sto_document.get("appendices", [])):
            if appendix.get("content_text"):
                appendix["content_text"] = self._rewrite_text_with_nvidia(
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
        self.rewrite_official_style_with_nvidia(sto_document)
        self.validate_schema(sto_document)
        return {"sto_document_json": sto_document, "transform_report": self.report}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform mapped STO JSON with NVIDIA Build API")
    parser.add_argument("--input-json", required=True, type=Path, help="Mapper output JSON path")
    parser.add_argument("--rules", required=True, type=Path, help="mapping-rules.yaml path")
    parser.add_argument("--schema", default=Path("sto-model.schema.json"), type=Path, help="Schema JSON path")
    parser.add_argument("--output-json", required=True, type=Path, help="Output transformed JSON path")
    parser.add_argument("--nvidia-api-key-env", default="NVIDIA_API_KEY", help="Environment variable containing NVIDIA API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL, MODEL_405B], help="NVIDIA model id")
    parser.add_argument("--api-url", default=DEFAULT_NVIDIA_URL, help="NVIDIA Build API URL")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    api_key = os.getenv(args.nvidia_api_key_env, "")
    if not api_key:
        print(
            json.dumps(
                {"error": f"Environment variable {args.nvidia_api_key_env} is required for real NVIDIA API mode"},
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
