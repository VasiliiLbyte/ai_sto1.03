# STOFabric

STOFabric converts draft organizational documents (`.docx`) into STO-compliant output for `СТО 31025229-001-2024`.

## Prerequisites

- Python 3.11+
- Windows PowerShell (examples below use PowerShell syntax)
- OpenRouter API key for full mode (`OPENROUTER_API_KEY`)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Environment Setup

1. Copy template:

```bash
copy .env.example .env
```

2. Fill at least:
- `OPENROUTER_API_KEY` for full pipeline (transformer stage)
- `STO_INPUT_DOCX` and `STO_OUTPUT_DOCX` for default controller runs

The project auto-loads `.env` in `controller.py`, `quality_controller.py`, and `content_transformer.py`.

Configuration priority is:

1. CLI arguments
2. Environment variables (`.env`)
3. Hardcoded defaults

## Quick Start

### Smoke (without LLM transformation)

```bash
python controller.py --input-docx "C:\path\to\draft.docx" --output-docx "_tmp_extract\smoke_out.docx" --mode up-to-mapper --save-intermediate --verbose
```

### Full run (with OpenRouter Gemini model)

```bash
python controller.py --input-docx "C:\path\to\draft.docx" --output-docx "_tmp_extract\full_out.docx" --mode full --model 70b --save-intermediate
```

Supported model keys:
- `8b` -> `google/gemini-2.5-flash`
- `70b` (default) -> `google/gemini-2.5-pro`
- `405b` -> alias to `google/gemini-2.5-pro` (legacy key)

### Quality controller: one draft

```bash
python quality_controller.py --test "Черновик Регламент заявки техники.docx" --model 70b --verbose
```

### Quality controller: all known drafts

```bash
python quality_controller.py --run-all --model 70b
```

## Expected Artifacts

- `_tmp_extract/mapper.json`
- `_tmp_extract/transformed.json` (full mode only)
- `_tmp_extract/quality_report/quality_report.json`
- `_tmp_extract/quality_report/quality_report.md`
- `final_sto.docx` in pipeline/quality output folders

## Typical Failures and Fast Diagnostics

- `OPENROUTER_API_KEY is required`:
  - Set `OPENROUTER_API_KEY` in `.env` or current shell.
- `... not found: <path>`:
  - Verify paths for input DOCX, rules, schema, checklist.
- `LLM API call failed after retries: timeout`:
  - Re-run later, test connectivity, try smaller input first, or run `--mode up-to-mapper`.
- `jsonschema is not installed` warning:
  - Install dependencies from `requirements.txt`.

## Migration Notes

- LLM config priority is: CLI -> `OPENROUTER_*` -> legacy `NVIDIA_*` -> defaults.
- `NVIDIA_*` variables are still accepted during migration, but considered deprecated.
- Prefer Gemini defaults:
  - `OPENROUTER_MODEL=google/gemini-2.5-pro`
  - `OPENROUTER_FALLBACK_MODEL=google/gemini-2.5-flash`

## Basic Validation

```bash
python -m py_compile parser.py semantic_mapper.py content_transformer.py formatter.py controller.py quality_controller.py
```
