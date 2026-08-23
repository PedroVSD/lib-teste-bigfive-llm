import os
from typing import Optional
from .base import LLMAdapter, AdapterConfig

class DeepSeekAdapter(LLMAdapter):
    """
    Adaptador exclusivo para a API oficial da DeepSeek.
    Utiliza a biblioteca 'openai' por baixo dos panos, mas aponta
    exclusivamente para a infraestrutura da DeepSeek.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)

        # A URL padrão oficial da DeepSeek
        default_url = "https://api.deepseek.com"

        # Tenta pegar do YAML, senão do .env, senão usa o padrão
        raw_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or default_url
        self.base_url = raw_url.rstrip("/")

        # Puxa a chave obrigatoriamente
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Falta a chave da API da DeepSeek! Defina 'DEEPSEEK_API_KEY' no ficheiro .env"
            )

        try:
            from openai import OpenAI
            # Instancia o cliente OpenAI mas forçando a URL da DeepSeek
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("Por favor instale a biblioteca openai: pip install openai")

    def complete(self, messages: list[dict], **kwargs) -> str:
        import time
        inicio = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        tempo = time.perf_counter() - inicio
        print(f"[{self.model}] OK | {tempo:.2f}s")
        return response.choices[0].message.content

    @property
    def identifier(self) -> str:
        return f"DeepSeekCloud:{self.model}"
