from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and common aliases to canonical keys."""
    raw = (value or "").strip().lower()
    if raw in {"openai", "gpt"}:
        return "openai"
    if raw in {"google", "gemini", "google-genai"}:
        return "gemini"
    if raw in {"anthropic", "anthorpic", "claude"}:
        return "anthropic"
    if raw in {"ollama", "local-ollama"}:
        return "ollama"
    if raw in {"openrouter", "open-router"}:
        return "openrouter"
    if raw in {"custom", "local", "vllm", "compatible"}:
        return "custom"
    return raw


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider.

    Supported providers:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenAI` configured for OpenRouter
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name or "gpt-4o-mini",
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.base_url or None,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or "EMPTY"
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL") or "http://localhost:8000/v1"
        return ChatOpenAI(
            model=config.model_name or "default",
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=config.model_name or "openai/gpt-4o-mini",
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name or "gemini-1.5-flash",
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model_name=config.model_name or "claude-3-5-sonnet-20241022",
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return ChatOllama(
            model=config.model_name or "llama3.2",
            temperature=config.temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported provider: '{config.provider}' (normalized: '{provider}')")
