from torch import Tensor
from model.Tokenizer import Tokenizer
from logging import Logger
from util.CustomLogger import CustomLogger
from transformers.models.gpt2.tokenization_gpt2 import (
    GPT2Tokenizer as BaseGPT2Tokenizer,
)


log: Logger = CustomLogger(__name__).get_logger()


class GPT2Tokenizer(Tokenizer):
    wrappedTokenizer: BaseGPT2Tokenizer

    def __init__(self) -> None:
        super().__init__()
        self.wrappedTokenizer = BaseGPT2Tokenizer.from_pretrained("gpt2")

    def encode(self, message: str) -> Tensor:
        log.debug(f"encode: {message}")
        encoded: Tensor = self.wrappedTokenizer.encode(
            message + "\n", return_tensors="pt"
        )  # type: ignore
        log.debug(f"encoded: {encoded}")
        return encoded

    def decode(self, tokenized: Tensor) -> str:
        log.debug(f"decode: {tokenized}")
        decoded: str = self.wrappedTokenizer.decode(tokenized)
        log.debug(f"decoded: {decoded}")
        return decoded  # type: ignore

    def getPadTokenID(self) -> int | None:
        return self.wrappedTokenizer.pad_token_id

    def getWrappedTokenizer(self) -> BaseGPT2Tokenizer:
        return self.wrappedTokenizer
