from typing_extensions import Self
import config.Config as Config
from torch import Tensor
from util.Channel import Channel
from model.Tokenizer import Tokenizer
from model.LanguageModel import LanguageModel
from twitchio import Message
import random


class OutputBuilder:
    channel: Channel
    prompt: Message | None
    model: LanguageModel
    tokenizer: Tokenizer
    input: Tensor

    def setChannel(self, channel: Channel) -> Self:
        self.channel = channel
        return self

    def setPrompt(self, prompt: Message) -> Self:
        self.prompt = prompt
        return self

    def setModel(self, model: LanguageModel) -> Self:
        self.model = model
        return self

    def setTokenizer(self, tokenizer: Tokenizer) -> Self:
        self.tokenizer = tokenizer
        return self

    def setInput(self, input: Tensor | None = None) -> Self:
        if input is not None:
            self.input = input
        return self

    async def build(self) -> str:
        return await self.generate_until_valid()

    async def generate_until_valid(self) -> str:
        i: int = 0
        while i < 50:
            i += 1
            generated: Tensor = self.model.generate(self.input)
            decoded: str = self.tokenizer.decode(generated).replace("\n", "").strip()
            if self.is_output_valid(decoded):
                return decoded

        return Config.ERROR_MESSAGE_COULD_NOT_GENERATE

    def roll(self, p: float) -> bool:
        return random.uniform(0, 1) < p

    def is_output_valid(self, line: str) -> bool:
        return (
            len(line) > 0
            and (" " in line or self.roll(0.5))
            and (len(line) >= 3 or self.roll(0.25))
            and (len(line) <= Config.OUTPUT_MAX_LENGTH or self.roll(0.1))
            and ("@" not in line or self.roll(0.05))
            and ("fishmoley" not in line.lower() or self.roll(0.1))
        )


class OutputCleaner:
    def clean_output(self, output: str) -> str | None:
        split: list[str] = output.splitlines()
        return self.get_first_acceptable_line(split)

    def get_first_acceptable_line(self, lines: list[str]) -> str | None:
        for line in lines:
            if self.is_acceptable(line):
                return line

    def roll(self, p: float) -> bool:
        return random.uniform(0, 1) < p

    def is_acceptable(self, line: str) -> bool:
        """Soft-filter for desirable conditions. Will let lines that don't
        meet the conditions through with some probability."""
        return (
            len(line) > 0
            and (" " in line or self.roll(0.5))
            and (len(line) >= 3 or self.roll(0.25))
            and (len(line) <= Config.OUTPUT_MAX_LENGTH or self.roll(0.1))
            and ("@" not in line or self.roll(0.05))
            and ("fishmoley" not in line.lower() or self.roll(0.1))
        )
