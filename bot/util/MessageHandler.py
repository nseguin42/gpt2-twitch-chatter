import logging
from util.Channel import Channel, ChannelList
from model.Tokenizer import Tokenizer
from model.LanguageModel import LanguageModel
from twitchio import Message
from util.MessageIgnorer import MessageIgnorer
from util.Pipeline import Pipeline
from util.CustomLogger import CustomLogger
import os

log: logging.Logger = CustomLogger(__name__).get_logger()


class MessageHandler:
    """
    Receives input from a chat server, tokenizes it, passes it to the model,
    then sends a reply if one is generated.
    """

    model: LanguageModel
    tokenizer: Tokenizer
    channels: ChannelList = ChannelList()

    def __init__(self, model: LanguageModel, tokenizer: Tokenizer) -> None:
        self.model: LanguageModel = model
        self.tokenizer: Tokenizer = tokenizer

    async def handle_message(self, message: Message) -> None:
        if MessageIgnorer(message).should_ignore():
            return

        log.debug(f"(#{message.channel.name}) {message.author.name}: {message.content}")

        channel: Channel = self.channels.get_channel(message.channel.name)
        await channel.add_message(message)

        if self.should_reply(message):
            await self.handle_reply(message, channel)

    async def handle_reply(self, message: Message, channel: Channel) -> None:
        response: str | None = await Pipeline(
            self.model, self.tokenizer, channel
        ).reply(message)
        if response is not None:
            log.info(f"({channel}) Replying to {message.author.name}: {response}")
            await message.channel.send(response)

    def should_reply(self, message: Message) -> bool:
        if not message.content:
            return False
        return os.getenv("NICK").lower() in message.content.lower()
