import os
import google.generativeai as genai
from typing import Optional
from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class GeminiAdapter(LLMAdapter):
    """
    Adapter para integrar os modelos Google Gemini à simulação de negociação.
    """

    def __init__(
        self, 
        model: str = "gemini-1.5-flash", 
        api_key: Optional[str] = None, 
        config: Optional[AdapterConfig] = None
    ):
        super().__init__(model, config)
        
        # Configura a chave de API (via parâmetro ou variável de ambiente)
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key do Gemini não fornecida. Defina GEMINI_API_KEY.")
        
        genai.configure(api_key=key)

    def complete(self, messages: list[dict], **kwargs) -> str:
        # 1. Separar o System Prompt (contexto e persona) do histórico de conversa
        system_instruction = None
        history = []
        
        for msg in messages:
            if msg["role"] == "system":
                # O Gemini trata o system prompt na inicialização do modelo
                system_instruction = msg["content"]
            else:
                # Traduzir papéis: "assistant" vira "model" no ecossistema Gemini
                role = "user" if msg["role"] == "user" else "model"
                history.append({"role": role, "parts": [msg["content"]]})
        
        # 2. Configurar os parâmetros de inferência (temperatura, etc)
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            **self.config.extra
        )
        
        # 3. Instanciar o modelo com as instruções de sistema
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_instruction
        )
        
        # 4. Enviar a requisição. O Gemini gerencia conversas via start_chat()
        try:
            if len(history) == 1:
                # É o primeiro turno (apenas uma mensagem do usuário)
                response = model.generate_content(
                    history[0]["parts"], 
                    generation_config=generation_config
                )
            else:
                # Há histórico: removemos a última mensagem para enviar agora
                last_msg = history.pop()
                chat = model.start_chat(history=history)
                response = chat.send_message(
                    last_msg["parts"], 
                    generation_config=generation_config
                )
                
            return response.text
            
        except Exception as e:
            # Útil para debugar limites de taxa (Rate Limits) ou erros na API
            print(f"[GeminiAdapter Error] Falha ao comunicar com a API: {e}")
            raise

    @property
    def identifier(self) -> str:
        return f"Gemini:{self.model}"