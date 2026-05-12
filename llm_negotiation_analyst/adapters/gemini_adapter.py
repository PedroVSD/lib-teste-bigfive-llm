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

        # 1. Mapeando e limpando o histórico para o formato estrito do Gemini
        for msg in messages:
            if msg["role"] == "system":
                # Se o motor enviar múltiplos system prompts, nós os concatenamos
                if system_instruction:
                    system_instruction += f"\n\n{msg['content']}"
                else:
                    system_instruction = msg["content"]
                continue

            role = "user" if msg["role"] == "user" else "model"

            # O Gemini NÃO aceita duas mensagens seguidas com o mesmo papel.
            # Se o papel atual for igual ao anterior, concatenamos o texto na mesma mensagem.
            if contents and contents[-1].role == role:
                contents[-1].parts[0].text += f"\n\n{msg['content']}"
            else:
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        # 2. A Regra de Ouro: A última mensagem DEVE ser do 'user'.
        if not contents:
            # Se a lista está vazia (só tinha system prompt), forçamos o início
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Inicie a negociação.")]))
        elif contents[-1].role == "model":
            # Se o motor terminou o histórico com 'model', passamos a bola de volta
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Continue a negociação e faça sua jogada.")]))

        GEMINI_PARAMS_VALIDOS = {
            "top_p", "top_k", "candidate_count", "stop_sequences",
            "presence_penalty", "frequency_penalty", "response_mime_type",
        }
        extra_filtrado = {k: v for k, v in self.config.extra.items() if k in GEMINI_PARAMS_VALIDOS}

        # 3. Configurando os parâmetros
        config_args = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            **extra_filtrado
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_args)

        # 4. Chamada da API com Resiliência
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generation_config
                )
                return response.text

            except Exception as e:
                error_str = str(e)
                # Verifica tanto limite de requisições (429) quanto servidor ocupado (503)
                if ("429" in error_str or "503" in error_str) and attempt < max_attempts - 1:
                    wait = 40 * (attempt + 1)  # Espera 40s, 80s, 120s
                    print(f"\n[GeminiAdapter] Rate limit ou sobrecarga (tentativa {attempt + 1}/{max_attempts}). Aguardando {wait}s...\n")
                    time.sleep(wait)
                else:
                    print(f"\n[GeminiAdapter Error] Falha fatal ao comunicar com a API: {e}\n")
                    raise

    @property
    def identifier(self) -> str:
        return f"Gemini:{self.model}"
