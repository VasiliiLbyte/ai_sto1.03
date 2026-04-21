from __future__ import annotations

from pathlib import Path

from content_transformer import _build_cli
from controller import _resolve_llm_api_key


def test_resolve_llm_api_key_prefers_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "legacy-key")

    key, using_legacy = _resolve_llm_api_key()

    assert key == "openrouter-key"
    assert using_legacy is False


def test_resolve_llm_api_key_uses_legacy_when_needed(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "legacy-key")

    key, using_legacy = _resolve_llm_api_key()

    assert key == "legacy-key"
    assert using_legacy is True


def test_resolve_llm_api_key_openrouter_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    key, using_legacy = _resolve_llm_api_key()

    assert key == "openrouter-key"
    assert using_legacy is False


def test_content_transformer_cli_prefers_openrouter_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    monkeypatch.setenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    monkeypatch.setenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

    parser = _build_cli()
    args = parser.parse_args(
        [
            "--input-json",
            str(tmp_path / "in.json"),
            "--rules",
            str(tmp_path / "rules.yaml"),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert args.model == "google/gemini-2.5-pro"
    assert args.api_url == "https://openrouter.ai/api/v1/chat/completions"
