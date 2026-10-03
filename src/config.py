from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Configuration settings for the memory systems lab.

    Attributes:
        base_dir: Root repository directory.
        data_dir: Directory containing benchmark datasets.
        state_dir: Directory where runtime state (e.g. User.md) is persisted.
        compact_threshold_tokens: Token count threshold to trigger compaction.
        compact_keep_messages: Number of recent messages to preserve during compaction.
        model: Configuration for the primary chat model.
        judge_model: Configuration for evaluation judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int = 600
    compact_keep_messages: int = 4
    model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    )
    judge_model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables and defaults.

    Args:
        base_dir: Optional base directory path. Defaults to repo root.

    Returns:
        Populated LabConfig instance.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Provider knobs
    primary_provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    primary_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    primary_temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API keys & base URLs per provider
    api_key: str | None = None
    base_url: str | None = None

    if primary_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif primary_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif primary_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif primary_provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif primary_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif primary_provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY", "EMPTY")
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")

    main_model_config = ProviderConfig(
        provider=primary_provider,
        model_name=primary_model,
        temperature=primary_temp,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model knobs
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", primary_provider))
    judge_model_name = os.getenv("JUDGE_MODEL", primary_model)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)

    judge_model_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=base_url if judge_provider == primary_provider else None,
    )

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=main_model_config,
        judge_model=judge_model_config,
    )
