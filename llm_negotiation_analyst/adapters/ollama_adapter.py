from typing import Optional
from .base import LLMAdapter, AdapterConfig


class OllamaAdapter(LLMAdapter):
    """
    Adapter for local models via Ollama (https://ollama.com).

    Ollama exposes an OpenAI-compatible REST API, so this adapter uses
    httpx directly to avoid requiring the openai package for local runs.

    Usage:
        adapter = OllamaAdapter(model="llama3.1:8b")
        reply = adapter.complete([{"role": "user", "content": "Hello"}])

    Requirements:
        - Ollama running locally: `ollama serve`
        - Model pulled: `ollama pull llama3.1:8b`
        - pip install httpx
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)
        self.base_url = base_url.rstrip("/")
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

    def complete(self, messages: list[dict], **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra,
            },
        }
        response = self._httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @property
    def identifier(self) -> str:
        return f"Ollama:{self.model}@{self.base_url}"
