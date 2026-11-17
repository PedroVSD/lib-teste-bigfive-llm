from typing import Optional
from .base import LLMAdapter, AdapterConfig

class OllamaLocalAdapter(LLMAdapter):
    """
    EXCLUSIVE adapter for running Ollama on your own machine (localhost).
    No authentication support; ignores external URLs for security.
    """

    supports_structured_output = True

    def __init__(
        self,
        model: str,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)
        # Fixed (hardcoded) to the default local Ollama port
        self.base_url = "http://localhost:11434"

        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("Install the httpx library: pip install httpx")

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._complete(messages, structured=False)

    def complete_structured(self, messages: list[dict], schema=None) -> str:
        return self._complete(messages, structured=True)

    def _complete(self, messages: list[dict], structured: bool) -> str:
        import time
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
        if structured:
            payload["format"] = "json"
        inicio = time.perf_counter()
        response = self._httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        tempo = time.perf_counter() - inicio
        if not self.config.quiet:
            print(f"[{self.model}] Status {response.status_code} | {tempo:.2f}s")
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        return content if content is not None else ""

    @property
    def identifier(self) -> str:
        return f"OllamaLocal:{self.model}"
