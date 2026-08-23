import os
import time
from typing import Optional
from google import genai
from google.genai import types

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class GeminiAdapter(LLMAdapter):
    """
    Adapter moderno para os modelos Google Gemini usando o novo SDK (google-genai).
    Inclui sistema de retentativas para contornar limites de taxa (429) e sobrecargas (503).
    """

    def __init__(
        self,
        model: str = "",
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None
    ):
        super().__init__(model, config)

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key do Gemini não fornecida. Defina GEMINI_API_KEY.")

        # O novo SDK usa um client instanciado
        self.client = genai.Client(api_key=key)

    def complete(self, messages: list[dict], **kwargs) -> str:
        system_instruction = None
        contents = []

        # -------------------------------------------------------------------------
        # Converte mensagens para o formato do Gemini
        # -------------------------------------------------------------------------
        for msg in messages:

            if msg["role"] == "system":
                if system_instruction:
                    system_instruction += f"\n\n{msg['content']}"
                else:
                    system_instruction = msg["content"]
                continue

            role = "user" if msg["role"] == "user" else "model"

            if contents and contents[-1].role == role:
                contents[-1].parts[0].text += f"\n\n{msg['content']}"
            else:
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Inicie a negociação.")]
                )
            )

        elif contents[-1].role == "model":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Continue a negociação e faça sua jogada."
                    )]
                )
            )

        # -------------------------------------------------------------------------
        # Configuração
        # -------------------------------------------------------------------------
        GEMINI_PARAMS_VALIDOS = {
            "top_p",
            "top_k",
            "candidate_count",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "response_mime_type",
        }

        extra_filtrado = {
            k: v
            for k, v in self.config.extra.items()
            if k in GEMINI_PARAMS_VALIDOS
        }

        config_args = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            **extra_filtrado,
        }

        if system_instruction:
            config_args["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_args)

        # -------------------------------------------------------------------------
        # Requisição
        # -------------------------------------------------------------------------
        max_attempts = 4

        for attempt in range(max_attempts):

            inicio = time.perf_counter()

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generation_config,
                )

                tempo = time.perf_counter() - inicio
                print(f"[{self.model}] OK | {tempo:.2f}s")
                # response.text pode ser None em caso de bloqueio/safety filter
                text = getattr(response, "text", None)
                if text is None:
                    # Tenta extrair de candidates
                    try:
                        if response.candidates and response.candidates[0].content.parts:
                            text = "".join(p.text or "" for p in response.candidates[0].content.parts)
                    except Exception:
                        text = None
                if text is None:
                    text = ""
                return text

            except Exception as e:

                tempo = time.perf_counter() - inicio
                error = str(e)

                if "429" in error:
                    print(f"[{self.model}] 429 Rate Limit ({tempo:.2f}s)")
                    if attempt < max_attempts - 1:
                        wait = 40 * (attempt + 1)
                        print(f"[{self.model}] Retry {attempt+1}/{max_attempts} em {wait}s")
                        time.sleep(wait)
                        continue
                if "503" in error:
                    print(f"[{self.model}] 503 Indisponível ({tempo:.2f}s)")
                    if attempt < max_attempts - 1:
                        wait = 40 * (attempt + 1)
                        print(f"[{self.model}] Retry {attempt+1}/{max_attempts} em {wait}s")
                        time.sleep(wait)
                        continue
                if "timeout" in error.lower():
                    raise RuntimeError(f"O modelo '{self.model}' excedeu o tempo limite.") from e
                raise RuntimeError(f"Erro Gemini: {error}") from e

        raise RuntimeError(
            f"Falha após {max_attempts} tentativas."
        )

    @property
    def identifier(self) -> str:
        return f"Gemini:{self.model}"
