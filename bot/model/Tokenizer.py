from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Any

from util.CustomLogger import CustomLogger


log: Logger = CustomLogger(__name__).get_logger()


class Tokenizer(metaclass=ABCMeta):
    """Abstract class for a tokenizer. Contains abstract methods that must be
    overridden by the concrete tokenizer."""

    wrappedTokenizer = Any

    def __init__(self) -> None:
        log.debug("init")

    @abstractmethod
    def encode(self, message: str) -> Any:
        pass

    @abstractmethod
    def decode(self, tokenized: Any) -> str:
        pass

    @abstractmethod
    def getWrappedTokenizer(self) -> Any:
        pass

    def get_newline_token(self) -> Any:
        return self.encode("\n")
