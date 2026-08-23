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
                print(f"[{self.model}] OK | {tempo:.2f}s")
                return response.choices[0].message.content
            except openai.APITimeoutError as e:
                tempo = time.perf_counter() - inicio
                if attempt < max_attempts - 1:
                    wait = 10 * (attempt + 1)
                    print(f"[{self.model}] TIMEOUT {tempo:.2f}s | retry {attempt+1}/{max_attempts} em {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"O modelo '{self.model}' excedeu o tempo limite de {self.config.timeout}s.") from e
            except openai.APIConnectionError as e:
                if attempt < max_attempts - 1:
                    wait = 10 * (attempt + 1)
                    print(f"[{self.model}] CONEXÃO falhou | retry {attempt+1}/{max_attempts} em {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError("Não foi possível conectar ao servidor do LM Studio.") from e
            except openai.APIStatusError as e:
                print(f"[{self.model}] HTTP {e.status_code}")
                raise RuntimeError(f"LM Studio retornou HTTP {e.status_code}.") from e
            except Exception:
                import traceback
                traceback.print_exc()
                raise

    @property
    def identifier(self) -> str:
        return f"LMStudio:{self.model}@{self.base_url}"
