from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import json

from ..guards import UTTERANCE_SCHEMA


@dataclass
class AdapterConfig:
    temperature: float = 0.0
    max_tokens: int = 4096  # aumentado para evitar respostas cortadas (era 1024)
    timeout: float = 300.0
    extra: dict = field(default_factory=dict)
    quiet: bool = False  # True = skip per-call status line (engine already prints the turn block)



class LLMAdapter(ABC):
    """
    Abstract base class for all LLM adapters.

    Every adapter must implement `complete`, which receives a list of
    OpenAI-style message dicts and returns the assistant reply as a string.

    Example message list:
        [
            {"role": "system", "content": "You are a buyer..."},
            {"role": "user",   "content": "I offer $100."},
        ]
    """

    # Native JSON mode support. Adapters that can force structured output set
    # this to True and implement complete_structured().
    supports_structured_output = False

    def __init__(self, model: str, config: Optional[AdapterConfig] = None):
        self.model = model
        self.config = config or AdapterConfig()

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages and return the model's reply."""
        ...

    def complete_structured(
        self,
        messages: list[dict],
        schema: Optional[dict] = None,
    ) -> str:
        """Ask the model for a structured utterance.

        The default implementation relies on the output-contract instruction
        already present in the system message. Adapters with a native JSON mode
        override this method to enforce the format at the provider level.
        """
        return self.complete(messages, response_schema=schema or UTTERANCE_SCHEMA)

    @property
    def identifier(self) -> str:
        """Human-readable identifier for logging and reports."""
        return f"{self.__class__.__name__}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self.identifier}>"

    def _debug_request(self, messages: list[dict], **extra) -> None:
        # Minimal log — detailed only when needed via logging DEBUG
        # Kept as no-op to avoid polluting the terminal (engine already logs turn/message)
        pass
