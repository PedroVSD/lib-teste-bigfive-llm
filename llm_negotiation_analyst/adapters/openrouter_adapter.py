import os
from typing import Optional
import openai

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig


class OpenRouterAdapter(LLMAdapter):
    """
    Adapter para OpenRouter (https://openrouter.ai) — API compatível com OpenAI.

    Suporta todos os modelos do OpenRouter, incluindo os gratuitos :free
    (ex: meta-llama/llama-3.1-8b-instruct:free, google/gemma-2-9b-it:free,
         mistralai/mistral-7b-instruct:free, qwen/qwen-2-7b-instruct:free).

    Requer OPENROUTER_API_KEY no .env ou via parâmetro. Opcionalmente define
    HTTP-Referer e X-Title para ranking no OpenRouter.
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "API key do OpenRouter não fornecida. "
                "Passe via parâmetro ou defina OPENROUTER_API_KEY no .env. "
                "Obtenha em https://openrouter.ai/keys"
            )

        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.referer = referer or os.environ.get("OPENROUTER_REFERER") or "https://github.com/llm-negotiation-analyst"
        self.title = title or os.environ.get("OPENROUTER_TITLE") or "llm-negotiation-analyst"

        # Cliente OpenAI apontado para OpenRouter
        self.client = openai.OpenAI(
            api_key=key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": self.referer,
                "X-Title": self.title,
            },
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        import time
        inicio = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra,
            )
            tempo = time.perf_counter() - inicio
            print(f"[{self.model}] OK | {tempo:.2f}s (OpenRouter)")
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            print(f"[{self.model}] ERRO OpenRouter: {e}")
            raise

    @property
    def identifier(self) -> str:
        return f"OpenRouter:{self.model}@{self.base_url}"
