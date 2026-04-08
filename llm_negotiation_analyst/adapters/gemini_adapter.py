import os
from typing import Optional
from google import genai
from google.genai import types

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class GeminiAdapter(LLMAdapter):
    """
    Adapter moderno para os modelos Google Gemini usando o novo SDK (google-genai).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
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

        # Mapeando o histórico para o formato estrito do novo SDK
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        # Configurando os parâmetros
        config_args = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            **self.config.extra
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config
            )
            return response.text

        except Exception as e:
            print(f"[GeminiAdapter Error] Falha ao comunicar com a API: {e}")
            raise

    @property
    def identifier(self) -> str:
        return f"Gemini:{self.model}"
