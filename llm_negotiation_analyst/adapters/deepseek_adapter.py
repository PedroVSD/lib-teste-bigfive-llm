import os
from typing import Optional
from .base import LLMAdapter, AdapterConfig

class DeepSeekAdapter(LLMAdapter):
    """
    Exclusive adapter for the official DeepSeek API.
    Uses the 'openai' library under the hood, but points
    exclusively at DeepSeek infrastructure.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__(model, config)

        # The official default DeepSeek URL
        default_url = "https://api.deepseek.com"

        # Try the YAML, then .env, then the default
        raw_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or default_url
        self.base_url = raw_url.rstrip("/")

        # API key is required
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Missing DeepSeek API key! Set 'DEEPSEEK_API_KEY' in the .env file"
            )

        try:
            from openai import OpenAI
            # Instantiate the OpenAI client forcing the DeepSeek URL
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("Please install the openai library: pip install openai")

    def complete(self, messages: list[dict], **kwargs) -> str:
        import time
        inicio = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        tempo = time.perf_counter() - inicio
        if not self.config.quiet:
            print(f"[{self.model}] OK | {tempo:.2f}s")
        return response.choices[0].message.content

    @property
    def identifier(self) -> str:
        return f"DeepSeekCloud:{self.model}"
