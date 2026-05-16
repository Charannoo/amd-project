"""Base agent — extend for custom agents."""
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    name = "BaseAgent"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError
