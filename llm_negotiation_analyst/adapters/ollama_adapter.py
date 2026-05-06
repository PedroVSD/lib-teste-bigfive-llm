import os
from typing import Optional
from .base import LLMAdapter, AdapterConfig

class OllamaAdapter(LLMAdapter):
    """
    Adapter for local and cloud models via Ollama.

    Ollama exposes an OpenAI-compatible REST API, so this adapter uses
    httpx directly. It supports optional authentication for cloud endpoints,
    but works perfectly without keys for local instances.
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
            raise ValueError("Falta a URL da API! Defina 'base_url' no YAML ou 'OLLAMA_BASE_URL' no arquivo .env")

        self.base_url = raw_url.rstrip("/")

        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")

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

        # Configurando os cabeçalhos de autenticação apenas se a chave existir (Nuvem)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._httpx.post(
            self.base_url,
            json=payload,
            headers=headers, # Se for local, o headers vai vazio {}. Se for nuvem, vai com a chave!
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @property
    def identifier(self) -> str:
        # Mostra no log se está usando autenticação ou não
        auth_status = "Auth" if self.api_key else "Local/NoAuth"
        return f"Ollama:{self.model}@{self.base_url}({auth_status})"
