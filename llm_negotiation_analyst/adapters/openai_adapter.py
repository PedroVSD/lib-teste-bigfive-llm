import os
from typing import Optional
import openai

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class OpenAIAdapter(LLMAdapter):
    """
    Adapter to integrate OpenAI models (GPT-4o, GPT-4-turbo, GPT-3.5)
    into the negotiation simulation.
    """

    supports_structured_output = True

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None
    ):
        super().__init__(model, config)

        # Look for the key in params or environment variables
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Pass it as a parameter or set the OPENAI_API_KEY environment variable."
            )

        # Instantiate the official OpenAI client
        self.client = openai.OpenAI(api_key=key)

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._complete(messages, structured=False)

    def complete_structured(self, messages: list[dict], schema=None) -> str:
        return self._complete(messages, structured=True)

    def _complete(self, messages: list[dict], structured: bool) -> str:
        import time
        inicio = time.perf_counter()
        extra = dict(self.config.extra)
        if structured:
            extra.setdefault("response_format", {"type": "json_object"})
        extra.setdefault("timeout", self.config.timeout)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **extra
            )
            tempo = time.perf_counter() - inicio
            if not self.config.quiet:
                print(f"[{self.model}] OK | {tempo:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{self.model}] ERROR: {e}")
            raise

    @property
    def identifier(self) -> str:
        return f"OpenAI:{self.model}"
