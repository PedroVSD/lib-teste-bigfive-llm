import os
from typing import Optional

from .base import LLMAdapter, AdapterConfig


_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaAdapter(LLMAdapter):
    """
    Adapter for NVIDIA NIM (https://build.nvidia.com), OpenAI-compatible API.

    Requires NVIDIA_API_KEY in .env or as a parameter. The base URL defaults to
    https://integrate.api.nvidia.com/v1 and can be overridden with NVIDIA_BASE_URL
    or the `base_url` parameter.

    `json_mode` enables the native `response_format={"type": "json_object"}`
    request. Set it to False for models/endpoints that reject that parameter;
    the output contract in the prompt still requests JSON.
    """

    supports_structured_output = True

    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        json_mode: bool = True,
    ):
        super().__init__(model, config)

        raw_url = base_url or os.environ.get("NVIDIA_BASE_URL") or _DEFAULT_BASE_URL
        self.base_url = raw_url.rstrip("/")

        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing NVIDIA API key! Set 'NVIDIA_API_KEY' in the .env file "
                "or pass api_key explicitly."
            )

        self.json_mode = json_mode

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai library: pip install openai")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._complete(messages, structured=False)

    def complete_structured(self, messages: list[dict], schema=None) -> str:
        return self._complete(messages, structured=True)

    def _complete(self, messages: list[dict], structured: bool) -> str:
        import time

        inicio = time.perf_counter()
        extra = dict(self.config.extra)
        if structured and self.json_mode:
            extra.setdefault("response_format", {"type": "json_object"})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **extra,
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
        return f"NVIDIA:{self.model}"
