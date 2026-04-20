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
        self.weights = self.rules["scoring_model"]["weights"]
        self.thresholds = self.rules["scoring_model"]["thresholds"]
        self.tie_break_order = self.rules["scoring_model"]["tie_break_order"]

    @staticmethod
    def load_mapping_rules(path: str | Path) -> dict[str, Any]:
        """Load YAML mapping rules."""
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def initialize_output_model() -> dict[str, Any]:
        """Create schema-compatible baseline model."""
        return {
            "meta": {
                "sto_number": "СТО 31025229-000-2024",
                "title": "Черновик СТО (автогенерация)",
                "version": "draft",
                "approval_date": "1970-01-01",
                "status": "draft",
                "extra_attributes": {"service_sheets": {}},
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
        """Classify service-sheet heading from rules."""
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

    def _extract_section_text(self, block: dict[str, Any]) -> str:
        return (block.get("normalized_text") or "").strip()

    def _match_section_rules(self, block: dict[str, Any]) -> list[Candidate]:
        """Generate section candidates based on section_mapping rules."""
        text = (block.get("normalized_text") or "").lower()
        candidates: list[Candidate] = []
        for rule in self.rules.get("section_mapping", []):
            patterns = [p.lower() for p in rule.get("source_patterns", [])]
            heading_match = 1.0 if any(p in text for p in patterns) else 0.0
            lexical_markers = 0.0
            lex_cfg = rule.get("lexical_markers", {})
            positives = [p.lower() for p in lex_cfg.get("positive", [])]
            negatives = [p.lower() for p in lex_cfg.get("negative", [])]
            if positives:
                hits = sum(1 for p in positives if p in text)
                lexical_markers += min(1.0, hits / max(1, len(positives)))
            if negatives:
                misses = sum(1 for p in negatives if p in text)
                lexical_markers -= min(0.7, misses / max(1, len(negatives)))

            if heading_match <= 0 and lexical_markers <= 0:
                continue
            candidates.append(
                Candidate(
                    target=rule["target"],
                    rule_id=rule.get("id", "unknown"),
                    heading_match=max(0.0, heading_match),
                    lexical_markers=max(0.0, lexical_markers),
                    positional_context=0.5,
                    structural_context=0.5 if block.get("heading_level") else 0.3,
                    schema_fit=1.0,
                )
            )
        return candidates

    def _match_explicit_blocks(self, block: dict[str, Any]) -> list[Candidate]:
        """Generate candidates for non-section specific areas."""
        text = (block.get("normalized_text") or "").lower()
        out: list[Candidate] = []

        for cfg_key in ("responsibility_mapping", "reporting_documents_mapping", "appendices_mapping"):
            for rule in self.rules.get(cfg_key, []):
                patterns = [p.lower() for p in rule.get("source_patterns", [])]
                if any(p in text for p in patterns):
                    out.append(
                        Candidate(
                            target=rule["target"],
                            rule_id=rule.get("id", cfg_key),
                            heading_match=1.0,
                            lexical_markers=0.8,
                            positional_context=0.5,
                            structural_context=0.5 if block.get("heading_level") else 0.3,
                            schema_fit=1.0,
                        )
                    )
        return out

    def generate_candidates(self, block: dict[str, Any]) -> list[Candidate]:
        """Generate all mapping candidates for a block."""
        if block.get("block_type") == "table":
            return [
                Candidate(
                    target="content.main.tables",
                    rule_id="TABLE-DEFAULT",
                    heading_match=0.2,
                    lexical_markers=0.2,
                    positional_context=0.5,
                    structural_context=0.8,
                    schema_fit=1.0,
                )
            ]
        return self._match_section_rules(block) + self._match_explicit_blocks(block)

    def score_candidate(self, candidate: Candidate) -> float:
        """Score candidate using weights from mapping rules."""
        score = (
            self.weights["heading_match"] * candidate.heading_match
            + self.weights["lexical_markers"] * candidate.lexical_markers
            + self.weights["positional_context"] * candidate.positional_context
            + self.weights["structural_context"] * candidate.structural_context
            + self.weights["schema_fit"] * candidate.schema_fit
        )
        candidate.score = round(score, 4)
        return candidate.score

    def resolve_candidate_conflict(self, candidates: list[Candidate]) -> Candidate | None:
        """Resolve best candidate by score and tie-break rules."""
        if not candidates:
            return None
        for cand in candidates:
            self.score_candidate(cand)
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        top = sorted_candidates[0]
        if len(sorted_candidates) == 1:
            return top

        tied = [c for c in sorted_candidates if abs(c.score - top.score) < 1e-6]
        if len(tied) == 1:
            return top

        for key in self.tie_break_order:
            if key == "explicit_heading_match":
                tied.sort(key=lambda c: c.heading_match, reverse=True)
            elif key == "schema_fit":
                tied.sort(key=lambda c: c.schema_fit, reverse=True)
            elif key == "semantic_similarity":
                tied.sort(key=lambda c: c.lexical_markers, reverse=True)
            top_candidate = tied[0]
            if len(tied) == 1 or tied[0].score != tied[1].score:
                return top_candidate
        return tied[0]

    def build_main_section_tree(self, blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Build recursive section tree with stack-based algorithm."""
        warnings: list[str] = []
        roots: list[dict[str, Any]] = []
        stack: list[tuple[int, dict[str, Any]]] = []
        code_seen: set[str] = set()

        stop_markers = [
            re.compile(self.rules["regex_library"]["appendix_header"], re.IGNORECASE),
            re.compile(r"лист регистрации изменений|лист ознакомления|лист согласования|согласовано", re.IGNORECASE),
        ]

        def push_section(section: dict[str, Any], depth: int) -> None:
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if stack:
                stack[-1][1]["subsections"].append(section)
            else:
                roots.append(section)
            stack.append((depth, section))

        for block in blocks:
            text = block.get("normalized_text", "")
            if any(rx.search(text) for rx in stop_markers):
                break
            heading_level = block.get("heading_level")
            if not heading_level:
                if stack and text:
                    current = stack[-1][1]
                    current["content"] = (current["content"] + "\n" + text).strip()
                    if block.get("block_type") == "table" and block.get("table_grid"):
                        current["tables"].append(block["table_grid"])
                    if block.get("image_ref"):
                        current["images"].extend(block["image_ref"])
                continue

            sec_number = block.get("numbering_token") or f"auto_{block['block_id']}"
            if sec_number in code_seen:
                sec_number = f"{sec_number}#2"
                warnings.append(f"RSD-W01 duplicate section number remapped: {sec_number}")
            code_seen.add(sec_number)
            section = {
                "section_number": sec_number,
                "title": text,
                "content": "",
                "subsections": [],
                "tables": [],
                "images": [],
                "bullet_points": [],
            }
            push_section(section, int(heading_level))
        return roots, warnings

    def _ensure_appendix(self, output: dict[str, Any], appendix_id: str, title: str) -> None:
        exists = next((a for a in output["appendices"] if a["appendix_id"] == appendix_id), None)
        if exists:
            return
        output["appendices"].append(
            {
                "appendix_id": appendix_id,
                "title": title,
                "status": "справочное",
                "content_text": "",
                "tables": [],
                "images": [],
            }
        )

    def project_block_to_target(
        self,
        block: dict[str, Any],
        selected: Candidate | None,
        output: dict[str, Any],
    ) -> None:
        """Project mapped block into output model."""
        text = self._extract_section_text(block)
        if not text and block.get("block_type") != "table":
            return

        if selected is None:
            output["meta"].setdefault("extra_attributes", {}).setdefault("unclassified_blocks", []).append(
                {
                    "block_id": block["block_id"],
                    "raw_text": block.get("raw_text", ""),
                    "normalized_text": text,
                }
            )
            return

        target = selected.target
        if target == "content.area":
            if not output["content"]["area"]:
                output["content"]["area"] = text
            else:
                output["content"]["area"] += "\n" + text
        elif target == "content.normative_references":
            output["content"]["normative_references"].append(
                {
                    "reference_id": text[:70] or "UNKNOWN",
                    "title": block.get("raw_text", text)[:300] or "Нормативный документ",
                }
            )
        elif target == "content.terms":
            term_match = re.match(r"^\s*([^:]{2,120})\s*:\s*(.{3,})$", text)
            abbr_match = re.match(r"^\s*([A-ZА-ЯЁ0-9]{2,15})\s*[-–]\s*(.{2,})$", text)
            if abbr_match:
                output["content"]["terms"]["abbreviations"].append(
                    {"abbr": abbr_match.group(1), "definition": abbr_match.group(2)}
                )
            elif term_match:
                output["content"]["terms"]["entries"].append(
                    {"term": term_match.group(1), "definition": term_match.group(2)}
                )
            elif not output["content"]["terms"]["intro"]:
                output["content"]["terms"]["intro"] = text
        elif target == "responsibility":
            output["responsibility"]["entries"].append({"role": text[:120], "responsibilities": [text]})
        elif target == "reporting_documents":
            output["reporting_documents"].append(
                {
                    "document_name": text[:180] or "Документ",
                    "responsible_role": "Не определено",
                    "storage_location": "Не определено",
                    "retention_period": "Не определено",
                }
            )
        elif target == "appendices":
            appendix_match = re.search(self.rules["regex_library"]["appendix_header"], block.get("raw_text", ""), re.IGNORECASE)
            appendix_id = appendix_match.group(1) if appendix_match else "A"
            self._ensure_appendix(output, appendix_id, text or f"Приложение {appendix_id}")
        elif target.startswith("content.main"):
            # Section tree is built separately in recursive pass.
            pass

    def post_validate_output(self, output: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Apply quality gates from rules."""
        warnings: list[str] = []
        errors: list[str] = []
        service_data = output["meta"].get("extra_attributes", {}).get("service_sheets", {})
        if output["content"]["main"]:
            for section in output["content"]["main"]:
                title = section.get("title", "").lower()
                if any(marker in title for marker in ("лист регистрации изменений", "лист ознакомления", "лист согласования")):
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
        if service_data and not isinstance(service_data, dict):
            warnings.append("Service sheet extract has unexpected format")
        return warnings, errors

    def map_blocks(self, parsed_doc: dict[str, Any]) -> dict[str, Any]:
        """Run end-to-end mapping."""
        output = self.initialize_output_model()
        mapping_trace: list[dict[str, Any]] = []
        warnings: list[str] = []
        errors: list[str] = []

        blocks = parsed_doc.get("blocks", [])
        service_sections: dict[str, list[dict[str, Any]]] = {"change_log": [], "acquaintance": [], "approval": []}
        main_candidates: list[dict[str, Any]] = []

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
                    }
                )
                continue

            candidates = self.generate_candidates(block)
            selected = self.resolve_candidate_conflict(candidates)
            confidence = selected.score if selected else 0.0
            if confidence < self.thresholds["review"]:
                warnings.append(f"Low confidence mapping block={block['block_id']} score={confidence:.2f}")
            self.project_block_to_target(block, selected, output)

            if selected and selected.target.startswith("content.main"):
                main_candidates.append(block)

            mapping_trace.append(
                {
                    "block_id": block["block_id"],
                    "source_text_excerpt": (block.get("normalized_text") or "")[:180],
                    "candidate_targets": [c.target for c in candidates],
                    "final_target": selected.target if selected else "unclassified",
                    "confidence": confidence,
                    "rule_ids": [selected.rule_id] if selected else [],
                }
            )

        main_tree, main_warnings = self.build_main_section_tree(main_candidates)
        output["content"]["main"] = main_tree
        warnings.extend(main_warnings)
        output["meta"]["extra_attributes"]["service_sheets"] = service_sections

        val_warnings, val_errors = self.post_validate_output(output)
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
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    parsed_doc = parse_document(args.docx_path)
    rules = SemanticMapper.load_mapping_rules(args.rules)
    mapper = SemanticMapper(rules)
    result = mapper.map_blocks(parsed_doc)

    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
