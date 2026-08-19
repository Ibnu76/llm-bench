"""Model and provider configuration."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Model:
    """LLM model configuration for benchmarking."""

    name: str
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0

    def __post_init__(self):
        provider_urls = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "together": "https://api.together.xyz/v1",
            "groq": "https://api.groq.com/openai/v1",
        }
        if not self.base_url and self.provider in provider_urls:
            self.base_url = provider_urls[self.provider]

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.provider})"
