"""Parser Agent for STOFabric.

Parses DOCX drafts into robust normalized blocks for semantic mapping.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

try:
    import docx2txt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    docx2txt = None


_WS_RE = re.compile(r"\s+")
_NUM_H1_RE = re.compile(r"^\s*(\d{1,2})\s+.+$")
_NUM_H2_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,3})\s+.+$")
_NUM_H3_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,3}\.\d{1,3})\s+.+$")
_NUM_TOKEN_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,2})\b")
_IMG_REL_RE = re.compile(r"/(image\d+\.[a-zA-Z0-9]+)$")
_FOOTNOTE_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


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
    """Load DOCX safely and raise explicit diagnostics."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DOCX file not found: {p}")
    try:
        return Document(str(p))
    except PackageNotFoundError as exc:
        raise ValueError(f"Invalid DOCX package: {p}") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Corrupted DOCX archive: {p}") from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unexpected DOCX read error: {p}") from exc


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
    """Extract section numbering token from heading-like lines."""
    match = _NUM_TOKEN_RE.match(text)
    return match.group(1) if match else None


def parse_table(table_obj: Table, table_idx: int, page_index: int) -> dict[str, Any]:
    """Extract table structure into parser-compatible grid."""
    rows: list[dict[str, Any]] = []
    columns_count = 0
    for r_idx, row in enumerate(table_obj.rows):
        cells = [normalize_text(cell.text) for cell in row.cells]
        columns_count = max(columns_count, len(cells))
        rows.append({"row_id": f"r{r_idx + 1}", "cells": cells})
    return {
        "table_id": f"table_{table_idx}",
        "title": f"Table {table_idx}",
        "columns_count": columns_count,
        "rows": rows,
        "source_page": page_index,
    }


def _alignment_to_name(alignment: Any) -> str | None:
    if alignment is None:
        return None
    raw = str(alignment)
    if "." in raw:
        return raw.split(".")[-1].lower()
    return raw.lower()


def _extract_style_flags(paragraph: Paragraph) -> dict[str, Any]:
    font_names = sorted({run.font.name for run in paragraph.runs if run.font and run.font.name})
    font_sizes = sorted(
        {float(run.font.size.pt) for run in paragraph.runs if run.font and run.font.size and run.font.size.pt}
    )
    return {
        "style_name": paragraph.style.name if paragraph.style else None,
        "is_bold_any": any(bool(run.bold) for run in paragraph.runs if run.text),
        "is_italic_any": any(bool(run.italic) for run in paragraph.runs if run.text),
        "is_underline_any": any(bool(run.underline) for run in paragraph.runs if run.text),
        "alignment": _alignment_to_name(paragraph.alignment),
        "font_names": font_names,
        "font_sizes": font_sizes,
    }


def _extract_image_refs_from_paragraph(paragraph: Paragraph, block_id: str) -> list[dict[str, Any]]:
    """Extract image references from paragraph runs with relation resolution."""
    image_refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_idx, run in enumerate(paragraph.runs):
        drawing_elems = run._element.xpath(".//*[local-name()='blip']")  # noqa: SLF001
        for draw_idx, drawing in enumerate(drawing_elems):
            rel_id = drawing.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if not rel_id or rel_id in seen:
                continue
            seen.add(rel_id)
            partname = None
            filename = None
            try:
                rel = paragraph.part.related_parts.get(rel_id)
                if rel is not None:
                    partname = str(getattr(rel, "partname", ""))
                    match = _IMG_REL_RE.search(partname or "")
                    if match:
                        filename = match.group(1)
            except Exception:
                partname = None
            image_refs.append(
                {
                    "image_id": f"{block_id}_img_{run_idx}_{draw_idx}",
                    "relationship_id": rel_id,
                    "partname": partname,
                    "filename": filename,
                }
            )
    return image_refs


def _has_page_break(paragraph: Paragraph) -> bool:
    xml = paragraph._element.xml  # noqa: SLF001
    return 'w:type="page"' in xml or "<w:lastRenderedPageBreak" in xml


def _has_section_break(paragraph: Paragraph) -> bool:
    xml = paragraph._element.xml  # noqa: SLF001
    return "<w:sectPr" in xml


def iter_docx_blocks(document: DocxDocument) -> Iterator[tuple[str, Paragraph | Table]]:
    """Iterate paragraph/table blocks in source order."""
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)


def _extract_header_footer_blocks(document: DocxDocument, start_index: int) -> tuple[list[ParsedBlock], int]:
    blocks: list[ParsedBlock] = []
    idx = start_index
    for section_idx, section in enumerate(document.sections):
        for kind, container in (("header", section.header), ("footer", section.footer)):
            for p_idx, paragraph in enumerate(container.paragraphs):
                raw = paragraph.text or ""
                normalized = normalize_text(raw)
                if not normalized:
                    continue
                block_id = f"b{idx:05d}"
                idx += 1
                blocks.append(
                    ParsedBlock(
                        block_id=block_id,
                        page_index=1,
                        block_index=idx,
                        line_range=[1, 1],
                        block_type=kind,
                        raw_text=raw,
                        normalized_text=normalized,
                        heading_level=None,
                        numbering_token=None,
                        bbox_or_position={"section_index": section_idx, "container_index": p_idx},
                        style_flags={"style_name": paragraph.style.name if paragraph.style else None},
                    )
                )
    return blocks, idx


def _extract_footnote_blocks(docx_path: Path, start_index: int) -> tuple[list[ParsedBlock], int]:
    blocks: list[ParsedBlock] = []
    idx = start_index
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/footnotes.xml" not in zf.namelist():
                return blocks, idx
            root = ET.fromstring(zf.read("word/footnotes.xml"))
            for node_idx, note in enumerate(root.findall(".//w:footnote", _FOOTNOTE_NS)):
                chunks: list[str] = []
                for t in note.findall(".//w:t", _FOOTNOTE_NS):
                    if t.text:
                        chunks.append(t.text)
                raw = " ".join(chunks).strip()
                normalized = normalize_text(raw)
                if not normalized:
                    continue
                block_id = f"b{idx:05d}"
                idx += 1
                blocks.append(
                    ParsedBlock(
                        block_id=block_id,
                        page_index=1,
                        block_index=idx,
                        line_range=[1, 1],
                        block_type="footnote",
                        raw_text=raw,
                        normalized_text=normalized,
                        heading_level=None,
                        numbering_token=None,
                        bbox_or_position={"footnote_index": node_idx},
                    )
                )
    except Exception:
        return blocks, idx
    return blocks, idx


def parse_document(docx_path: str | Path) -> dict[str, Any]:
    """Parse document into normalized blocks expected by semantic mapper."""
    path = Path(docx_path)
    diagnostics: list[str] = []
    metadata: dict[str, Any] = {
        "source_document": str(path),
        "page_index_mode": "layout_estimate",
        "page_index_confidence": "low",
    }
    if docx2txt is not None:
        metadata["page_index_mode"] = "layout_estimate_plus_docx2txt"
        metadata["page_index_confidence"] = "medium"
    else:
        diagnostics.append("docx2txt is not installed; using page break heuristic only")

    document = load_docx(path)
    blocks: list[ParsedBlock] = []
    table_counter = 0
    current_page = 1
    block_serial = 1

    for block_index, (kind, payload) in enumerate(iter_docx_blocks(document)):
        block_id = f"b{block_serial:05d}"
        block_serial += 1

        if kind == "paragraph":
            paragraph = payload
            raw_text = paragraph.text or ""
            normalized = normalize_text(raw_text)
            style_flags = _extract_style_flags(paragraph)
            heading_level = detect_heading_level(normalized, style_flags.get("style_name"))
            numbering_token = extract_numbering_token(normalized)
            image_refs = _extract_image_refs_from_paragraph(paragraph, block_id)
            block_type = "heading" if heading_level else "paragraph"
            if image_refs and not normalized:
                block_type = "image"
            block = ParsedBlock(
                block_id=block_id,
                page_index=current_page,
                block_index=block_index,
                line_range=[1, 1],
                block_type=block_type,
                raw_text=raw_text,
                normalized_text=normalized,
                heading_level=heading_level,
                numbering_token=numbering_token,
                bbox_or_position={"block_index": block_index, "source": "body"},
                style_flags=style_flags,
                image_ref=image_refs or None,
            )
            blocks.append(block)
            if _has_page_break(paragraph) or _has_section_break(paragraph):
                current_page += 1
            continue

        table = payload
        table_counter += 1
        table_grid = parse_table(table, table_counter, current_page)
        raw_text = "\n".join(" | ".join(row["cells"]) for row in table_grid["rows"])
        block = ParsedBlock(
            block_id=block_id,
            page_index=current_page,
            block_index=block_index,
            line_range=[1, 1],
            block_type="table",
            raw_text=raw_text,
            normalized_text=normalize_text(raw_text),
            heading_level=None,
            numbering_token=None,
            bbox_or_position={"block_index": block_index, "source": "body"},
            table_grid=table_grid,
        )
        blocks.append(block)

    header_footer_blocks, block_serial = _extract_header_footer_blocks(document, block_serial)
    blocks.extend(header_footer_blocks)
    footnote_blocks, block_serial = _extract_footnote_blocks(path, block_serial)
    blocks.extend(footnote_blocks)

    metadata["blocks_count"] = len(blocks)
    metadata["diagnostics"] = diagnostics
    return {"metadata": metadata, "blocks": [asdict(block) for block in blocks]}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse DOCX draft into normalized blocks.")
    parser.add_argument("docx_path", type=Path, help="Path to input .docx file")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--preview", type=int, default=10, help="Preview first N blocks in stdout")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    try:
        parsed = parse_document(args.docx_path)
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    if args.output:
        args.output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = parsed["blocks"][: args.preview]
    print(json.dumps({"metadata": parsed["metadata"], "preview_blocks": preview}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
