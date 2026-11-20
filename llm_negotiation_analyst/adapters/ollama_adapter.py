import os
import time
import httpx

from typing import Optional

from .base import LLMAdapter, AdapterConfig


class OllamaAdapter(LLMAdapter):
    """
    Adapter for local and cloud models via Ollama.
    """

    supports_structured_output = True

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

        raw_url = raw_url.rstrip("/")
        if raw_url.endswith("/api/generate"):
            # This adapter speaks the chat API (messages -> message.content).
            # /api/generate uses prompt -> response and would return an empty
            # message, silently breaking every call.
            fixed = raw_url[: -len("/api/generate")] + "/api/chat"
            if not self.config.quiet:
                print(
                    f"[Ollama] base_url '{raw_url}' is the generate endpoint; "
                    f"using '{fixed}' instead."
                )
            raw_url = fixed
        self.base_url = raw_url
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._complete(messages, structured=False)

    def complete_structured(self, messages: list[dict], schema=None) -> str:
        return self._complete(messages, structured=True)

    def _complete(self, messages: list[dict], structured: bool) -> str:
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
            # Ollama's JSON mode forces a valid JSON object; the system prompt
            # defines the required {"utterance": "..."} shape.
            payload["format"] = "json"

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

            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Ollama API error: {data['error']}")

            content = self._extract_content(data)
            if not content:
                message = data.get("message") or {}
                print(
                    f"[{self.model}] EMPTY content | "
                    f"done_reason={data.get('done_reason')} | "
                    f"thinking={bool(message.get('thinking'))} | "
                    f"keys={list(data)[:8]}"
                )
            return content

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

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Accept native chat, generate and OpenAI-compatible response shapes."""
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content

        native = data.get("response")
        if isinstance(native, str) and native:
            return native

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            try:
                return choices[0]["message"]["content"] or ""
            except (KeyError, TypeError, IndexError):
                pass

        return ""

    @property
    def identifier(self) -> str:
        auth_status = "Auth" if self.api_key else "Local/NoAuth"
        return f"Ollama:{self.model}@{self.base_url}({auth_status})"
