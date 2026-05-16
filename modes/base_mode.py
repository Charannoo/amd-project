"""Base mode interface."""
from abc import ABC, abstractmethod
from typing import Any


class BaseMode(ABC):
    @abstractmethod
    def run(self, *args: Any, tag: str = "", **kwargs: Any) -> dict:
        raise NotImplementedError
