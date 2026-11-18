import os
from typing import Optional

import httpx

from .base import LLMAdapter, AdapterConfig


_DEFAULT_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


class CloudflareAdapter(LLMAdapter):
    """
    Adapter for Cloudflare Workers AI (https://developers.cloudflare.com/workers-ai/).

    Uses the REST endpoint:
        POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}

    Requires CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in .env (or the
    `api_token` / `account_id` parameters). Model IDs are the catalog names,
    e.g. "@cf/meta/llama-3.1-8b-instruct".

    `json_mode` enables Workers AI JSON Mode
    (`response_format={"type": "json_object"}`). Only some models support it;
    set it to False for the others. The output contract in the prompt still
    requests JSON either way.
    """

    supports_structured_output = True

    def __init__(
        self,
        model: str = "@cf/meta/llama-3.1-8b-instruct",
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        json_mode: bool = True,
    ):
        super().__init__(model, config)

        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError(
                "Missing Cloudflare Account ID! Set 'CLOUDFLARE_ACCOUNT_ID' in "
                "the .env file or pass account_id explicitly."
            )

        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Missing Cloudflare API token! Set 'CLOUDFLARE_API_TOKEN' in "
                "the .env file or pass api_token explicitly."
            )

        raw_base = base_url or os.environ.get("CLOUDFLARE_API_BASE") or _DEFAULT_API_BASE
        self.base_url = raw_base.rstrip("/")

        self.json_mode = json_mode

        self._url = f"{self.base_url}/{self.account_id}/ai/run/{self.model}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._complete(messages, structured=False)

    def complete_structured(self, messages: list[dict], schema=None) -> str:
        return self._complete(messages, structured=True)

    def _complete(self, messages: list[dict], structured: bool) -> str:
        import json
        import time

        inicio = time.perf_counter()

        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if structured and self.json_mode:
            payload.setdefault("response_format", {"type": "json_object"})
        payload.update(self.config.extra)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self._url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[{self.model}] ERROR: {e}")
            raise

        if not data.get("success", True):
            errors = data.get("errors") or data
            print(f"[{self.model}] ERROR: {errors}")
            raise RuntimeError(f"Cloudflare Workers AI error: {errors}")

        text = self._extract_text(data)
        tempo = time.perf_counter() - inicio
        if not self.config.quiet:
            print(f"[{self.model}] OK | {tempo:.2f}s")
        return text

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Normalize the Workers AI response to a plain string."""
        import json

        result = data.get("result")

        if isinstance(result, dict):
            if "response" in result:
                inner = result["response"]
                # JSON Mode with json_schema returns an object
                if isinstance(inner, (dict, list)):
                    return json.dumps(inner, ensure_ascii=False)
                return "" if inner is None else str(inner)
            if "choices" in result:
                return result["choices"][0]["message"]["content"]

        if isinstance(result, str):
            return result

        # OpenAI-compatible shape (some endpoints wrap choices at the top level)
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return json.dumps(data, ensure_ascii=False)

    @property
    def identifier(self) -> str:
        return f"CloudflareWorkersAI:{self.model}"
