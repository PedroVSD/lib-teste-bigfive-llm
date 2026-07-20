import time
from typing import Optional

import openai

from llm_negotiation_analyst.adapters.base import (
    LLMAdapter,
    AdapterConfig,
)


class LMStudioAdapter(LLMAdapter):
    """
    Adapter para integrar modelos locais rodando no LM Studio.

    O LM Studio expõe uma API compatível com a OpenAI.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)

        self.base_url = base_url.rstrip("/")

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="lm-studio",
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        max_attempts = 3

        self._debug_request(
            messages,
            URL=self.base_url,
            Provider="LM Studio",
        )

        for attempt in range(max_attempts):

            inicio = time.perf_counter()

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    **self.config.extra,
                )

                tempo = time.perf_counter() - inicio

                print(f"Latência    : {tempo:.2f}s")
                print("Resposta recebida com sucesso.")
                print("=" * 80)

                return response.choices[0].message.content

            except openai.APITimeoutError as e:

                tempo = time.perf_counter() - inicio

                print("=" * 80)
                print("ERRO: Timeout")
                print(f"Modelo      : {self.model}")
                print(f"Tempo gasto : {tempo:.2f}s")
                print(f"Timeout cfg : {self.config.timeout}s")
                print("=" * 80)

                if attempt < max_attempts - 1:
                    wait = 10 * (attempt + 1)

                    print(
                        f"[LMStudioAdapter] Nova tentativa em "
                        f"{wait}s ({attempt + 1}/{max_attempts})"
                    )

                    time.sleep(wait)
                    continue

                raise RuntimeError(
                    f"O modelo '{self.model}' excedeu o tempo "
                    f"limite de {self.config.timeout}s."
                ) from e

            except openai.APIConnectionError as e:

                print("=" * 80)
                print("ERRO: Falha de conexão")
                print(e)
                print("=" * 80)

                if attempt < max_attempts - 1:
                    wait = 10 * (attempt + 1)

                    print(
                        f"[LMStudioAdapter] Tentando novamente "
                        f"em {wait}s..."
                    )

                    time.sleep(wait)
                    continue

                raise RuntimeError(
                    "Não foi possível conectar ao servidor do LM Studio."
                ) from e

            except openai.APIStatusError as e:

                print("=" * 80)
                print("ERRO HTTP")
                print(f"Status      : {e.status_code}")
                print(f"Resposta    : {e.response}")
                print("=" * 80)

                raise RuntimeError(
                    f"LM Studio retornou HTTP {e.status_code}."
                ) from e

            except Exception:

                print("=" * 80)
                print("ERRO INESPERADO")

                import traceback
                traceback.print_exc()

                print("=" * 80)

                raise

    @property
    def identifier(self) -> str:
        return f"LMStudio:{self.model}@{self.base_url}"
