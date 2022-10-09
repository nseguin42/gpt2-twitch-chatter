import os
from torch import Tensor
from util.Channel import Channel
from model.Tokenizer import Tokenizer
from model.LanguageModel import LanguageModel
from twitchio import Message
import config.Config as Config
from util.Output import OutputBuilder
from util.StringUtils import remove_self_mentions
from util.TensorBuilder import TensorBuilder
import logging
from util.CustomLogger import CustomLogger

log: logging.Logger = CustomLogger(__name__).get_logger()


class Pipeline:
    model: LanguageModel
    tokenizer: Tokenizer
    channel: Channel

    def __init__(
        self, model: LanguageModel, tokenizer: Tokenizer, channel: Channel
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.channel = channel

    async def reply(self, message: Message) -> str | None:
        log.info(f"({self.channel}) {message.author.name}: {message.content}")
        if self.channel.is_rate_limited():
            log.info("Rate limited, ignoring prompt.")
            return

        await self.channel.reset_limit()

        assert message.content
        text: str = remove_self_mentions(message.content)
        tokenizedMessage: Tensor = self.tokenizer.encode(text)

        channelHistory: Tensor = await self.channel.get_tokens(self.tokenizer)
        input: Tensor = (
            TensorBuilder()
            .append(channelHistory)
            .appendNTimes(tokenizedMessage, Config.PROMPT_DUPLICATION_FACTOR)
            .build()
        )

        return await self.generate(input)

    async def generate(self, input: Tensor | None = None) -> str:
        return (
            await OutputBuilder()
            .setModel(self.model)
            .setTokenizer(self.tokenizer)
            .setInput(input)
            .build()
        )
