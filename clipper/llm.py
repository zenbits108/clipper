"""Config-driven LLM provider clients used by select.py.

The provider is never hardcoded in calling code: a channel config names
"ollama" or "openrouter" and get_provider() resolves that name to a client
using connection details from config/settings.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import requests

from clipper.config import Settings


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""


class Provider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class OllamaProvider:
    base_url: str
    model: str
    timeout: float = 300.0

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        try:
            resp = requests.post(
                url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        response = data.get("response")
        if not response:
            raise ProviderError(f"Ollama returned no 'response' field: {data}")
        return response


@dataclass
class OpenRouterProvider:
    base_url: str
    model: str
    api_key: str
    timeout: float = 300.0

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected OpenRouter response shape: {data}") from exc


def get_provider(name: str, settings: Settings, model_override: Optional[str] = None) -> Provider:
    """Resolve a provider name (from channel config) to a client instance."""
    if name == "ollama":
        return OllamaProvider(
            base_url=settings.ollama.base_url,
            model=model_override or settings.ollama.default_model,
        )
    if name == "openrouter":
        api_key = os.environ.get(settings.openrouter.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Environment variable '{settings.openrouter.api_key_env}' is not set; "
                "required for the openrouter provider."
            )
        return OpenRouterProvider(
            base_url=settings.openrouter.base_url,
            model=model_override or settings.openrouter.default_model,
            api_key=api_key,
        )
    raise ProviderError(f"Unknown provider '{name}'")
