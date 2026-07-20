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
        # Logs
        # -------------------------------------------------------------------------
        self._debug_request(
            messages,
            Provider="Google Gemini",
        )

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

                print(f"Latência    : {tempo:.2f}s")
                print("Resposta recebida com sucesso.")
                print("=" * 80)

                return response.text

            except Exception as e:

                tempo = time.perf_counter() - inicio
                error = str(e)

                # -----------------------------------------------------------------
                # Rate limit
                # -----------------------------------------------------------------
                if "429" in error:

                    print("=" * 80)
                    print("ERRO: Rate Limit (429)")
                    print(f"Tempo gasto : {tempo:.2f}s")
                    print("=" * 80)

                    if attempt < max_attempts - 1:

                        wait = 40 * (attempt + 1)

                        print(
                            f"[GeminiAdapter] Nova tentativa em "
                            f"{wait}s ({attempt + 1}/{max_attempts})"
                        )

                        time.sleep(wait)
                        continue

                # -----------------------------------------------------------------
                # Sobrecarga
                # -----------------------------------------------------------------
                if "503" in error:

                    print("=" * 80)
                    print("ERRO: Serviço indisponível (503)")
                    print(f"Tempo gasto : {tempo:.2f}s")
                    print("=" * 80)

                    if attempt < max_attempts - 1:

                        wait = 40 * (attempt + 1)

                        print(
                            f"[GeminiAdapter] Nova tentativa em "
                            f"{wait}s ({attempt + 1}/{max_attempts})"
                        )

                        time.sleep(wait)
                        continue

                # -----------------------------------------------------------------
                # Timeout
                # -----------------------------------------------------------------
                if "timeout" in error.lower():

                    print("=" * 80)
                    print("ERRO: Timeout")
                    print(f"Tempo gasto : {tempo:.2f}s")
                    print("=" * 80)

                    raise RuntimeError(
                        f"O modelo '{self.model}' excedeu "
                        f"o tempo limite configurado."
                    ) from e

                # -----------------------------------------------------------------
                # Outros erros
                # -----------------------------------------------------------------
                print("=" * 80)
                print("ERRO INESPERADO")
                print(error)
                print("=" * 80)

                raise RuntimeError(
                    f"Erro ao comunicar com a API Gemini: {error}"
                ) from e

        raise RuntimeError(
            f"Falha após {max_attempts} tentativas."
        )

    @property
    def identifier(self) -> str:
        return f"Gemini:{self.model}"
