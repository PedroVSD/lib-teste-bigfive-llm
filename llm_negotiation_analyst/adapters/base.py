from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class AdapterConfig:
    temperature: float = 0.0
    max_tokens: int = 4096  # aumentado para evitar respostas cortadas (era 1024)
    timeout: float = 300.0
    extra: dict = field(default_factory=dict)



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

    def __init__(self, model: str, config: Optional[AdapterConfig] = None):
        self.model = model
        self.config = config or AdapterConfig()

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages and return the model's reply."""
        ...

    @property
    def identifier(self) -> str:
        """Human-readable identifier for logging and reports."""
        return f"{self.__class__.__name__}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self.identifier}>"

    def _debug_request(self, messages: list[dict], **extra) -> None:
        # Log minimalista — detalhado apenas se necessário via logging DEBUG
        # Mantido como no-op para não poluir o terminal (engine já loga turno/mensagem)
        pass
