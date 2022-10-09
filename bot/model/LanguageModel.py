from abc import abstractmethod
from typing import Any
from model.Tokenizer import Tokenizer
from abc import ABCMeta


class LanguageModel(metaclass=ABCMeta):
    """
    Abstract class for a language model. Contains abstract methods that must
    be overridden by the implementations.
    """

    tokenizer: Tokenizer
    model: Any

    @abstractmethod
    def generate(self, input: Any) -> Any:
        pass
