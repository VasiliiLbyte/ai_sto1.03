"""Semantic Mapper Agent for STOFabric.

Maps parser blocks into STO JSON model according to mapping-rules.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from parser import parse_document

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


_TERM_PAIR_RE = re.compile(r"^\s*([^:]{2,120})\s*:\s*(.{3,})$")
_ABBR_PAIR_RE = re.compile(r"^\s*([A-ZА-ЯЁ0-9]{2,15})\s*[-–]\s*(.{2,})$")
_NORM_REF_RE = re.compile(r"(ГОСТ|СТО|ISO|IEC|Приказ|ФЗ|РД)\s*[A-Za-zА-Яа-я0-9.\-/– ]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SERVICE_MARKER_RE = re.compile(r"лист регистрации изменений|лист ознакомления|лист согласования|согласовано", re.IGNORECASE)


@dataclass(slots=True)
class Candidate:
    """Mapping target candidate with feature scores."""

    target: str
    rule_id: str
    heading_match: float
    lexical_markers: float
    positional_context: float
    structural_context: float
    schema_fit: float
    score: float = 0.0


class SemanticMapper:
    """Rule-driven semantic mapper."""

    def __init__(self, rules: dict[str, Any]) -> None:
        self.rules = rules["mapping_rules"]
        scoring_model = self.rules.get("scoring_model", {})
        self.weights = scoring_model.get(
            "weights",
            {
                "heading_match": 0.35,
                "lexical_markers": 0.25,
                "positional_context": 0.15,
                "structural_context": 0.15,
                "schema_fit": 0.10,
            },
        )
        self.thresholds = scoring_model.get("thresholds", {"accept": 0.78, "review": 0.60, "reject": 0.40})
        self.tie_break_order = scoring_model.get(
            "tie_break_order",
            ["explicit_heading_match", "mandatory_section_priority", "schema_fit", "semantic_similarity"],
        )
        self.mandatory_priority = scoring_model.get("mandatory_section_priority", [])
        self.regex_library = self.rules.get("regex_library", {})

    @staticmethod
    def load_mapping_rules(path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            # Fallback for partially invalid YAML indentation in evolving rules files.
            fixed = text.replace("\n    conflict_resolution:\n", "\n  conflict_resolution:\n")
            parsed = yaml.safe_load(fixed)
            parsed.setdefault("mapping_rules", {}).setdefault("_load_diagnostics", []).append(
                "rules_loaded_with_fallback_indentation_fix"
            )
            return parsed

    @staticmethod
    def load_schema(path: str | Path | None) -> dict[str, Any] | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def initialize_output_model() -> dict[str, Any]:
        return {
            "meta": {
                "sto_number": "СТО 31025229-000-2024",
                "title": "Черновик СТО (автогенерация)",
                "version": "draft",
                "approval_date": "1970-01-01",
                "status": "draft",
                "extra_attributes": {"service_sheets": {}, "mapping_diagnostics": {}},
            },
            "content": {
                "area": "",
                "normative_references": [],
                "terms": {"intro": "", "entries": [], "abbreviations": []},
                "main": [],
                "document_flow_diagrams": [],
                "extra_attributes": {},
            },
            "reporting_documents": [],
            "responsibility": {"title": "Ответственность", "entries": []},
            "appendices": [],
        }

    def classify_service_sheet(self, block: dict[str, Any]) -> str | None:
        text = (block.get("normalized_text") or "").lower()
        for rule in self.rules.get("service_sheets_mapping", {}).get("detection_rules", []):
            for pattern in rule.get("source_patterns", []):
                if pattern.lower() in text:
                    target = rule.get("target", "")
                    if target.endswith("change_log"):
                        return "change_log"
                    if target.endswith("acquaintance"):
                        return "acquaintance"
                    if target.endswith("approval"):
                        return "approval"
        return None

    @staticmethod
    def _extract_section_text(block: dict[str, Any]) -> str:
        return (block.get("normalized_text") or "").strip()

    def _schema_fit_score(self, target: str) -> float:
        valid_prefixes = (
            "content.area",
            "content.normative_references",
            "content.terms",
            "content.main",
            "responsibility",
            "reporting_documents",
            "appendices",
        )
        return 1.0 if any(target.startswith(p) for p in valid_prefixes) else 0.2

    def _positional_context_score(self, block: dict[str, Any]) -> float:
        page = int(block.get("page_index") or 1)
        if page <= 2:
            return 0.75
        if page <= 5:
            return 0.55
        return 0.4

    def _match_rule_list(self, block: dict[str, Any], rules_key: str) -> list[Candidate]:
        text = (block.get("normalized_text") or "").lower()
        out: list[Candidate] = []
        for rule in self.rules.get(rules_key, []):
            patterns = [p.lower() for p in rule.get("source_patterns", [])]
            heading_match = 1.0 if any(p in text for p in patterns) else 0.0
            lexical_markers = 0.0
            lex_cfg = rule.get("lexical_markers", {})
            positives = [p.lower() for p in lex_cfg.get("positive", [])]
            negatives = [p.lower() for p in lex_cfg.get("negative", [])]
            if positives:
                lexical_markers += min(1.0, sum(1 for p in positives if p in text) / max(1, len(positives)))
            if negatives:
                lexical_markers -= min(0.7, sum(1 for p in negatives if p in text) / max(1, len(negatives)))
            if heading_match <= 0 and lexical_markers <= 0:
                continue
            out.append(
                Candidate(
                    target=rule["target"],
                    rule_id=rule.get("id", rules_key),
                    heading_match=max(0.0, heading_match),
                    lexical_markers=max(0.0, lexical_markers),
                    positional_context=self._positional_context_score(block),
                    structural_context=0.6 if block.get("heading_level") else 0.35,
                    schema_fit=self._schema_fit_score(rule["target"]),
                )
            )
        return out

    def _table_target_from_matrix(self, context_target: str | None, block: dict[str, Any]) -> str:
        text = (block.get("normalized_text") or "").lower()
        if context_target == "reporting_documents" or "отчет" in text or "отчёт" in text:
            return "reporting_documents"
        if context_target == "appendices" or "приложение" in text or "форма" in text:
            return "appendices"
        return "content.main.tables"

    def _image_target_from_matrix(self, context_target: str | None, block: dict[str, Any]) -> str:
        text = ((block.get("raw_text") or "") + " " + (block.get("normalized_text") or "")).lower()
        if context_target == "appendices":
            return "appendices"
        if any(word in text for word in ("блок-схема", "workflow", "алгоритм")):
            return "content.document_flow_diagrams"
        return "content.main.images"

    def generate_candidates(self, block: dict[str, Any], context_target: str | None) -> list[Candidate]:
        if block.get("block_type") == "table":
            target = self._table_target_from_matrix(context_target, block)
            return [
                Candidate(
                    target=target,
                    rule_id="TABLE-MATRIX",
                    heading_match=0.4,
                    lexical_markers=0.5,
                    positional_context=self._positional_context_score(block),
                    structural_context=0.8,
                    schema_fit=self._schema_fit_score(target),
                )
            ]
        if block.get("block_type") == "image":
            target = self._image_target_from_matrix(context_target, block)
            return [
                Candidate(
                    target=target,
                    rule_id="IMAGE-MATRIX",
                    heading_match=0.3,
                    lexical_markers=0.5,
                    positional_context=self._positional_context_score(block),
                    structural_context=0.8,
                    schema_fit=self._schema_fit_score(target),
                )
            ]
        return (
            self._match_rule_list(block, "section_mapping")
            + self._match_rule_list(block, "responsibility_mapping")
            + self._match_rule_list(block, "reporting_documents_mapping")
            + self._match_rule_list(block, "appendices_mapping")
            + self._match_rule_list(block, "meta_mapping")
        )

    def score_candidate(self, candidate: Candidate) -> float:
        score = (
            self.weights["heading_match"] * candidate.heading_match
            + self.weights["lexical_markers"] * candidate.lexical_markers
            + self.weights["positional_context"] * candidate.positional_context
            + self.weights["structural_context"] * candidate.structural_context
            + self.weights["schema_fit"] * candidate.schema_fit
        )
        candidate.score = round(score, 4)
        return candidate.score

    def resolve_candidate_conflict(self, candidates: list[Candidate]) -> tuple[Candidate | None, str]:
        if not candidates:
            return None, "no_candidates"
        for cand in candidates:
            self.score_candidate(cand)
        ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
        top = ordered[0]
        tied = [c for c in ordered if abs(c.score - top.score) < 1e-6]
        if len(tied) == 1:
            return top, "max_score"

        reason = "tie_break"
        for key in self.tie_break_order:
            if key == "explicit_heading_match":
                tied.sort(key=lambda c: c.heading_match, reverse=True)
            elif key == "mandatory_section_priority":
                tied.sort(
                    key=lambda c: self.mandatory_priority.index(c.target)
                    if c.target in self.mandatory_priority
                    else 999
                )
            elif key == "schema_fit":
                tied.sort(key=lambda c: c.schema_fit, reverse=True)
            elif key == "semantic_similarity":
                tied.sort(key=lambda c: c.lexical_markers, reverse=True)
            if len(tied) == 1:
                return tied[0], f"{reason}:{key}"
        return tied[0], f"{reason}:stable_first"

    def _normalize_section_number(self, token: str | None, block_id: str) -> tuple[str, str | None]:
        if not token:
            return f"auto_{block_id}", "RSD-W03 missing numbering token"
        fixed = token.replace("..", ".").strip(".")
        if fixed != token:
            return fixed, f"RSD-W02 malformed numbering normalized: {token}->{fixed}"
        return fixed, None

    def build_main_section_tree(self, blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        roots: list[dict[str, Any]] = []
        stack: list[tuple[int, dict[str, Any]]] = []
        seen_codes: set[str] = set()
        appendix_pattern = self.regex_library.get("appendix_header", r"^\s*Приложение\s+")
        appendix_re = re.compile(appendix_pattern, re.IGNORECASE)

        def push(section: dict[str, Any], depth: int) -> None:
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if stack:
                stack[-1][1]["subsections"].append(section)
            else:
                roots.append(section)
            stack.append((depth, section))

        prev_token: str | None = None
        for block in blocks:
            text = block.get("normalized_text", "")
            if _SERVICE_MARKER_RE.search(text) or appendix_re.search(text):
                break

            if not block.get("heading_level"):
                if stack and text:
                    cur = stack[-1][1]
                    cur["content"] = (cur["content"] + "\n" + text).strip()
                    if block.get("block_type") == "table" and block.get("table_grid"):
                        cur["tables"].append(block["table_grid"])
                    if block.get("image_ref"):
                        cur["images"].extend(block["image_ref"])
                continue

            sec_number, norm_warn = self._normalize_section_number(block.get("numbering_token"), block["block_id"])
            if norm_warn:
                warnings.append(norm_warn)

            if sec_number in seen_codes:
                sec_number = f"{sec_number}#2"
                warnings.append(f"RSD-W01 duplicate section number remapped: {sec_number}")
            seen_codes.add(sec_number)

            if prev_token and sec_number.count(".") == prev_token.count("."):
                try:
                    prev_last = int(prev_token.split(".")[-1].replace("#2", ""))
                    curr_last = int(sec_number.split(".")[-1].replace("#2", ""))
                    if curr_last - prev_last > 1:
                        warnings.append(f"RSD-W04 numbering jump detected: {prev_token}->{sec_number}")
                except Exception:
                    pass
            prev_token = sec_number

            section = {
                "section_number": sec_number,
                "title": text,
                "content": "",
                "subsections": [],
                "tables": [],
                "images": [],
                "bullet_points": [],
            }
            push(section, int(block["heading_level"]))
        return roots, warnings

    @staticmethod
    def _ensure_appendix(output: dict[str, Any], appendix_id: str, title: str, status: str = "справочное") -> dict[str, Any]:
        exists = next((a for a in output["appendices"] if a["appendix_id"] == appendix_id), None)
        if exists:
            return exists
        app = {
            "appendix_id": appendix_id,
            "title": title,
            "status": status if status in ("обязательное", "рекомендуемое", "справочное") else "справочное",
            "content_text": "",
            "tables": [],
            "images": [],
        }
        output["appendices"].append(app)
        return app

    def _project_normative_reference(self, output: dict[str, Any], text: str, raw_text: str) -> None:
        ref_match = _NORM_REF_RE.search(raw_text) or _NORM_REF_RE.search(text)
        year_match = _YEAR_RE.search(raw_text) or _YEAR_RE.search(text)
        reference_id = ref_match.group(0).strip() if ref_match else text[:70] or "UNKNOWN"
        title = raw_text.strip() or text or "Нормативный документ"
        ref_obj = {"reference_id": reference_id, "title": title[:500]}
        if year_match:
            ref_obj["year"] = int(year_match.group(0))
        output["content"]["normative_references"].append(ref_obj)

    def project_block_to_target(
        self,
        block: dict[str, Any],
        selected: Candidate | None,
        output: dict[str, Any],
    ) -> None:
        text = self._extract_section_text(block)
        raw_text = block.get("raw_text", "")
        if not text and block.get("block_type") not in ("table", "image"):
            return

        if selected is None:
            output["meta"]["extra_attributes"].setdefault("unclassified_blocks", []).append(
                {"block_id": block["block_id"], "raw_text": raw_text, "normalized_text": text}
            )
            return

        target = selected.target
        if target == "content.area":
            output["content"]["area"] = (output["content"]["area"] + "\n" + text).strip() if output["content"]["area"] else text
            return
        if target == "content.normative_references":
            self._project_normative_reference(output, text, raw_text)
            return
        if target == "content.terms":
            term_match = _TERM_PAIR_RE.match(text)
            abbr_match = _ABBR_PAIR_RE.match(text)
            if abbr_match:
                output["content"]["terms"]["abbreviations"].append(
                    {"abbr": abbr_match.group(1).strip(), "definition": abbr_match.group(2).strip()}
                )
            elif term_match:
                output["content"]["terms"]["entries"].append(
                    {"term": term_match.group(1).strip(), "definition": term_match.group(2).strip()}
                )
            elif not output["content"]["terms"]["intro"]:
                output["content"]["terms"]["intro"] = text
            else:
                output["content"]["terms"]["notes"] = (output["content"]["terms"].get("notes", "") + "\n" + text).strip()
            return
        if target.startswith("responsibility"):
            role = text.split(":", 1)[0][:120] if ":" in text else text[:120]
            output["responsibility"]["entries"].append({"role": role or "Не определено", "responsibilities": [text]})
            return
        if target.startswith("reporting_documents"):
            if block.get("block_type") == "table" and block.get("table_grid"):
                table = block["table_grid"]
                for row in table.get("rows", []):
                    cells = row.get("cells", [])
                    output["reporting_documents"].append(
                        {
                            "document_name": (cells[0] if len(cells) > 0 and cells[0] else "Документ"),
                            "responsible_role": (cells[1] if len(cells) > 1 and cells[1] else "Не определено"),
                            "storage_location": (cells[2] if len(cells) > 2 and cells[2] else "Не определено"),
                            "retention_period": (cells[3] if len(cells) > 3 and cells[3] else "Не определено"),
                        }
                    )
            else:
                output["reporting_documents"].append(
                    {
                        "document_name": text[:200] or "Документ",
                        "responsible_role": "Не определено",
                        "storage_location": "Не определено",
                        "retention_period": "Не определено",
                    }
                )
            return
        if target.startswith("appendices"):
            appendix_re = re.compile(self.regex_library.get("appendix_header", r"^\s*Приложение\s+([A-Za-zА-Яа-яЁё]{1,3})"), re.IGNORECASE)
            match = appendix_re.search(raw_text) or appendix_re.search(text)
            appendix_id = match.group(1) if match else "A"
            app = self._ensure_appendix(output, appendix_id, text or f"Приложение {appendix_id}")
            if block.get("block_type") == "table" and block.get("table_grid"):
                app["tables"].append(block["table_grid"])
            elif block.get("block_type") == "image" and block.get("image_ref"):
                app["images"].extend(block["image_ref"])
            else:
                app["content_text"] = (app["content_text"] + "\n" + text).strip() if app["content_text"] else text
            return
        if target == "content.document_flow_diagrams":
            for img in block.get("image_ref") or []:
                output["content"]["document_flow_diagrams"].append(
                    {
                        "image_id": img.get("image_id"),
                        "title": img.get("filename") or img.get("image_id") or "Схема",
                        "caption": raw_text or text,
                        "source_page": block.get("page_index"),
                    }
                )
            return
        if target.startswith("content.main"):
            output["content"]["extra_attributes"].setdefault("main_staging_blocks", []).append(block)
            return
        if target.startswith("meta."):
            output["meta"]["extra_attributes"].setdefault("meta_candidates", []).append({"target": target, "text": text})
            return
        output["meta"]["extra_attributes"].setdefault("unclassified_blocks", []).append(
            {"block_id": block["block_id"], "raw_text": raw_text, "normalized_text": text}
        )

    def _postprocess_service_exclusion(self, output: dict[str, Any]) -> None:
        filtered: list[dict[str, Any]] = []
        for section in output["content"]["main"]:
            title = (section.get("title") or "").lower()
            if _SERVICE_MARKER_RE.search(title):
                output["meta"]["extra_attributes"]["service_sheets"].setdefault("purged_from_main", []).append(section)
                continue
            filtered.append(section)
        output["content"]["main"] = filtered

    def validate_schema(self, sto_document: dict[str, Any], schema: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        if schema is None:
            warnings.append("Schema file is not available; schema validation skipped")
            return warnings, errors
        if jsonschema is None:
            warnings.append("jsonschema is not installed; strict schema validation skipped")
            return warnings, errors
        try:
            jsonschema.validate(sto_document, schema)
        except Exception as exc:
            errors.append(f"Schema validation failed: {exc}")
        return warnings, errors

    def post_validate_output(self, output: dict[str, Any], schema: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        if output["content"]["main"]:
            for section in output["content"]["main"]:
                title = section.get("title", "").lower()
                if _SERVICE_MARKER_RE.search(title):
                    errors.append("QG-01 service sheet found in content.main")
        if not output["content"]["area"] or not output["content"]["main"]:
            errors.append("QG-02 mandatory content sections are incomplete")
        for app in output["appendices"]:
            if not app.get("appendix_id") or not app.get("title") or not app.get("status"):
                errors.append("QG-03 appendix is missing id/title/status")
        for row in output["reporting_documents"]:
            required = ("document_name", "responsible_role", "storage_location", "retention_period")
            if any(not row.get(k) for k in required):
                warnings.append("QG-04 reporting document missing required fields")
        schema_w, schema_e = self.validate_schema(output, schema)
        warnings.extend(schema_w)
        errors.extend(schema_e)
        return warnings, errors

    def map_blocks(self, parsed_doc: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        output = self.initialize_output_model()
        mapping_trace: list[dict[str, Any]] = []
        warnings: list[str] = []
        errors: list[str] = []

        blocks = parsed_doc.get("blocks", [])
        service_sections: dict[str, list[dict[str, Any]]] = {"change_log": [], "acquaintance": [], "approval": []}
        main_candidates: list[dict[str, Any]] = []
        context_target: str | None = None

        for block in blocks:
            service_type = self.classify_service_sheet(block)
            if service_type:
                service_sections[service_type].append(block)
                mapping_trace.append(
                    {
                        "block_id": block["block_id"],
                        "candidate_targets": ["service_sheet"],
                        "final_target": f"meta.extra_attributes.service_sheets.{service_type}",
                        "confidence": 1.0,
                        "rule_ids": ["SRV"],
                        "decision_reason": "service_sheet_match",
                    }
                )
                continue

            candidates = self.generate_candidates(block, context_target)
            selected, reason = self.resolve_candidate_conflict(candidates)
            confidence = selected.score if selected else 0.0
            if confidence < self.thresholds["review"]:
                warnings.append(f"Low confidence mapping block={block['block_id']} score={confidence:.2f}")

            self.project_block_to_target(block, selected, output)
            if selected:
                context_target = selected.target
                if selected.target.startswith("content.main"):
                    main_candidates.append(block)
            elif confidence < self.thresholds["reject"]:
                output["meta"]["extra_attributes"].setdefault("unclassified_blocks", []).append(block)

            mapping_trace.append(
                {
                    "block_id": block["block_id"],
                    "source_text_excerpt": (block.get("normalized_text") or "")[:180],
                    "candidate_targets": [c.target for c in candidates],
                    "candidate_scores": [
                        {
                            "target": c.target,
                            "score": c.score,
                            "features": {
                                "heading_match": c.heading_match,
                                "lexical_markers": c.lexical_markers,
                                "positional_context": c.positional_context,
                                "structural_context": c.structural_context,
                                "schema_fit": c.schema_fit,
                            },
                        }
                        for c in candidates
                    ],
                    "final_target": selected.target if selected else "unclassified",
                    "confidence": confidence,
                    "rule_ids": [selected.rule_id] if selected else [],
                    "decision_reason": reason,
                }
            )

        main_tree, main_warnings = self.build_main_section_tree(main_candidates)
        output["content"]["main"] = main_tree
        warnings.extend(main_warnings)
        output["meta"]["extra_attributes"]["service_sheets"] = service_sections
        self._postprocess_service_exclusion(output)

        val_warnings, val_errors = self.post_validate_output(output, schema)
        warnings.extend(val_warnings)
        errors.extend(val_errors)

        return {
            "sto_document_json": output,
            "mapping_trace": mapping_trace,
            "warnings": warnings,
            "errors": errors,
            "service_sheet_extract": service_sections,
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run semantic mapper over DOCX draft.")
    parser.add_argument("docx_path", type=Path, help="Path to source DOCX")
    parser.add_argument("--rules", type=Path, default=Path("mapping-rules.yaml"), help="Mapping rules YAML path")
    parser.add_argument("--schema", type=Path, default=Path("sto-model.schema.json"), help="STO JSON schema path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    try:
        parsed_doc = parse_document(args.docx_path)
        rules = SemanticMapper.load_mapping_rules(args.rules)
        mapper = SemanticMapper(rules)
        schema = SemanticMapper.load_schema(args.schema)
        result = mapper.map_blocks(parsed_doc, schema=schema)
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "warnings": len(result.get("warnings", [])),
        "errors": len(result.get("errors", [])),
        "mapping_trace": len(result.get("mapping_trace", [])),
        "output_file": str(args.output) if args.output else None,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
