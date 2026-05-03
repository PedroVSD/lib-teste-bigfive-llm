import time
from typing import Optional
import openai

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class LMStudioAdapter(LLMAdapter):
    """
    Adapter para integrar modelos locais rodando no LM Studio.
    O LM Studio expõe um servidor REST compatível com a biblioteca da OpenAI.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        config: Optional[AdapterConfig] = None
    ):
        super().__init__(model, config)

        # A biblioteca da OpenAI exige uma chave, mas o LM Studio aceita qualquer texto
        dummy_key = "lm-studio"

        # Instancia o cliente apontando para a sua própria máquina
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=dummy_key
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **self.config.extra
                )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()

                # Trata possíveis gargalos da sua máquina (timeout ou recusa de conexão)
                if "connection" in error_str or "timeout" in error_str:
                    if attempt < max_attempts - 1:
                        wait = 10 * (attempt + 1)
                        print(f"\n[LMStudioAdapter] Servidor local falhou (tentativa {attempt + 1}/{max_attempts}). Aguardando {wait}s...\n")
                        time.sleep(wait)
                    else:
                        print(f"\n[LMStudioAdapter Error] Falha fatal de conexão local: {e}\n")
                        raise
                else:
                    print(f"\n[LMStudioAdapter Error] Erro inesperado: {e}\n")
                    raise

    @property
    def identifier(self) -> str:
        return f"LMStudio:{self.model}"
