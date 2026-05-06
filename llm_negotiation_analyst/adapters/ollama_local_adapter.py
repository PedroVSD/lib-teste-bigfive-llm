from typing import Optional
from .base import LLMAdapter, AdapterConfig

class OllamaLocalAdapter(LLMAdapter):
    """
    Adapter EXCLUSIVO para rodar o Ollama na sua própria máquina (localhost).
    Não possui suporte a autenticação e ignora URLs externas por segurança.
    """

    def __init__(
        self,
        model: str,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)
        # Fixado (hardcoded) para a porta padrão do Ollama local
        self.base_url = "http://localhost:11434"

        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("Instale a biblioteca httpx: pip install httpx")

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

        # Requisição direta, sem headers de autorização
        response = self._httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @property
    def identifier(self) -> str:
        return f"OllamaLocal:{self.model}"
