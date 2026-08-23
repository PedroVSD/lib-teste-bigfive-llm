import os
from typing import Optional
import openai

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class OpenAIAdapter(LLMAdapter):
    """
    Adapter para integrar os modelos da OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5)
    à simulação de negociação.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None
    ):
        super().__init__(model, config)

        # Busca a chave nos parâmetros ou nas variáveis de ambiente
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "API key da OpenAI não fornecida. "
                "Passe via parâmetro ou defina a variável de ambiente OPENAI_API_KEY."
            )

        # Instancia o cliente oficial da OpenAI
        self.client = openai.OpenAI(api_key=key)

    def complete(self, messages: list[dict], **kwargs) -> str:
        import time
        inicio = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra
            )
            tempo = time.perf_counter() - inicio
            print(f"[{self.model}] OK | {tempo:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{self.model}] ERRO: {e}")
            raise

    @property
    def identifier(self) -> str:
        return f"OpenAI:{self.model}"
