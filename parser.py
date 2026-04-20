"""Parser Agent for STOFabric.

Parses DOCX draft documents into normalized block structures for semantic mapping.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


_WS_RE = re.compile(r"\s+")
_NUM_H1_RE = re.compile(r"^\s*(\d{1,2})\s+.+$")
_NUM_H2_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,3})\s+.+$")
_NUM_H3_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,3}\.\d{1,3})\s+.+$")
_NUM_TOKEN_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,2})\b")


@dataclass(slots=True)
class ParsedBlock:
    """Normalized parser output block."""

    block_id: str
    page_index: int
    block_index: int
    line_range: list[int]
    block_type: str
    raw_text: str
    normalized_text: str
    heading_level: int | None
    numbering_token: str | None
    bbox_or_position: dict[str, Any]
    style_flags: dict[str, Any] = field(default_factory=dict)
    table_grid: dict[str, Any] | None = None
    image_ref: list[dict[str, Any]] | None = None
    caption_ref: str | None = None


def load_docx(path: str | Path) -> DocxDocument:
    """Load DOCX document."""
    return Document(str(path))


def normalize_text(text: str) -> str:
    """Normalize text for downstream lexical matching."""
    if not text:
        return ""
    normalized = text.replace("—", "-").replace("–", "-")
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def detect_heading_level(text: str, style_name: str | None = None) -> int | None:
    """Detect heading level from numbering regex and style hints."""
    if _NUM_H3_RE.match(text):
        return 3
    if _NUM_H2_RE.match(text):
        return 2
    if _NUM_H1_RE.match(text):
        return 1

    if style_name:
        style_l = style_name.lower()
        if "heading 1" in style_l:
            return 1
        if "heading 2" in style_l:
            return 2
        if "heading 3" in style_l:
            return 3
    return None


def extract_numbering_token(text: str) -> str | None:
    """Extract section numbering token from a heading-like line."""
    match = _NUM_TOKEN_RE.match(text)
    return match.group(1) if match else None


def parse_table(table_obj: Table, table_idx: int) -> dict[str, Any]:
    """Extract table structure into parser-compatible grid."""
    rows: list[dict[str, Any]] = []
    columns_count = 0
    for r_idx, row in enumerate(table_obj.rows):
        cells = [normalize_text(cell.text) for cell in row.cells]
        columns_count = max(columns_count, len(cells))
        rows.append(
            {
                "row_id": f"r{r_idx + 1}",
                "cells": cells,
            }
        )
    return {
        "table_id": f"table_{table_idx}",
        "title": f"Table {table_idx}",
        "columns_count": columns_count,
        "rows": rows,
    }


def _extract_image_refs_from_paragraph(paragraph: Paragraph, block_id: str) -> list[dict[str, Any]]:
    """Extract image relationship ids from paragraph runs."""
    image_refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_idx, run in enumerate(paragraph.runs):
        drawing_elems = run._element.xpath(".//*[local-name()='blip']")  # noqa: SLF001
        for d_idx, drawing in enumerate(drawing_elems):
            rel_id = drawing.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if not rel_id or rel_id in seen:
                continue
            seen.add(rel_id)
            image_refs.append(
                {
                    "image_id": f"{block_id}_img_{run_idx}_{d_idx}",
                    "relationship_id": rel_id,
                }
            )
    return image_refs


def iter_docx_blocks(document: DocxDocument) -> Iterator[tuple[str, Paragraph | Table]]:
    """Iterate paragraph/table blocks in source order."""
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)


def parse_document(docx_path: str | Path) -> dict[str, Any]:
    """Parse document into normalized blocks expected by semantic mapper."""
    document = load_docx(docx_path)
    blocks: list[ParsedBlock] = []
    table_counter = 0

    for block_index, (kind, payload) in enumerate(iter_docx_blocks(document)):
        block_id = f"b{block_index + 1:05d}"
        page_index = 1  # python-docx does not expose layout pages; keep deterministic placeholder.
        line_range = [1, 1]

        if kind == "paragraph":
            paragraph = payload
            raw_text = paragraph.text or ""
            normalized = normalize_text(raw_text)
            style_name = paragraph.style.name if paragraph.style else None
            heading_level = detect_heading_level(normalized, style_name)
            numbering_token = extract_numbering_token(normalized)
            image_refs = _extract_image_refs_from_paragraph(paragraph, block_id)
            block_type = "heading" if heading_level else "paragraph"
            style_flags = {
                "style_name": style_name,
                "is_bold_any": any(run.bold for run in paragraph.runs if run.text),
                "alignment": str(paragraph.alignment) if paragraph.alignment is not None else None,
            }
            block = ParsedBlock(
                block_id=block_id,
                page_index=page_index,
                block_index=block_index,
                line_range=line_range,
                block_type=block_type,
                raw_text=raw_text,
                normalized_text=normalized,
                heading_level=heading_level,
                numbering_token=numbering_token,
                bbox_or_position={"block_index": block_index},
                style_flags=style_flags,
                image_ref=image_refs or None,
            )
            blocks.append(block)
            continue

        table = payload
        table_counter += 1
        table_grid = parse_table(table, table_counter)
        raw_text = "\n".join(" | ".join(row["cells"]) for row in table_grid["rows"])
        block = ParsedBlock(
            block_id=block_id,
            page_index=page_index,
            block_index=block_index,
            line_range=line_range,
            block_type="table",
            raw_text=raw_text,
            normalized_text=normalize_text(raw_text),
            heading_level=None,
            numbering_token=None,
            bbox_or_position={"block_index": block_index},
            table_grid=table_grid,
        )
        blocks.append(block)

    return {
        "metadata": {
            "source_document": str(docx_path),
            "blocks_count": len(blocks),
        },
        "blocks": [asdict(block) for block in blocks],
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse DOCX draft into normalized blocks.")
    parser.add_argument("docx_path", type=Path, help="Path to input .docx file")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--preview", type=int, default=10, help="Preview first N blocks in stdout")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    parsed = parse_document(args.docx_path)

    if args.output:
        args.output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = parsed["blocks"][: args.preview]
    print(json.dumps({"metadata": parsed["metadata"], "preview_blocks": preview}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
