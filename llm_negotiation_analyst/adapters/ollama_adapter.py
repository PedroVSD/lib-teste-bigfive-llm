import os
import time
import httpx

from typing import Optional

from .base import LLMAdapter, AdapterConfig


class OllamaAdapter(LLMAdapter):
    """
    Adapter for local and cloud models via Ollama.
    """

    def __init__(
        self,
        model: str = "",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)

        raw_url = base_url or os.environ.get("OLLAMA_BASE_URL")

        if not raw_url:
            raise ValueError(
                "Falta a URL da API! Defina 'base_url' no YAML ou "
                "'OLLAMA_BASE_URL' no arquivo .env."
            )

        self.base_url = raw_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")

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

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        inicio = time.perf_counter()

        try:
            response = httpx.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )

            tempo = time.perf_counter() - inicio

            # Minimal log: status + model + latency (engine logs turn/message)
            if not self.config.quiet:
                print(f"[{self.model}] Status {response.status_code} | {tempo:.2f}s")

            response.raise_for_status()

            content = response.json().get("message", {}).get("content")
            return content if content is not None else ""

        except httpx.ReadTimeout as e:
            tempo = time.perf_counter() - inicio
            print(f"[{self.model}] TIMEOUT after {tempo:.2f}s (limit {self.config.timeout}s)")
            raise RuntimeError(
                f"Model '{self.model}' exceeded the "
                f"{self.config.timeout}s timeout."
            ) from e

        except httpx.HTTPStatusError as e:
            print(f"[{self.model}] HTTP {e.response.status_code}: {e.response.text[:500]}")
            raise

        except Exception:
            import traceback
            traceback.print_exc()
            raise

    @property
    def identifier(self) -> str:
        auth_status = "Auth" if self.api_key else "Local/NoAuth"
        return f"Ollama:{self.model}@{self.base_url}({auth_status})"
