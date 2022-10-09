from dataclasses import dataclass, field
from typing import List
from aiolimiter import AsyncLimiter
from twitchio import Message
from util.MessageQueue import *
import config.Config as Config
from util.CustomLogger import CustomLogger

log: logging.Logger = CustomLogger(__name__).get_logger()


@dataclass
class Channel:
    """Class representing a Twitch channel."""

    name: str
    messages: MessageQueue = field(default_factory=lambda: MessageQueue())
    limit: AsyncLimiter = field(
        default_factory=lambda: AsyncLimiter(1, Config.RATE_LIMIT_SECONDS)
    )

    def __str__(self) -> str:
        return f"#{self.name}"

    def is_rate_limited(self) -> bool:
        return not self.limit.has_capacity()

    def get_limit(self) -> AsyncLimiter:
        return self.limit

    async def reset_limit(self) -> None:
        await self.limit.acquire()

    async def add_message(self, message: Message) -> None:
        await self.messages.add_message(message)

    async def get_tokens(self, tokenizer: Tokenizer) -> Tensor:
        return await self.messages.tokenize(tokenizer)


class ChannelList(List[Channel]):
    def __init__(self) -> None:
        super().__init__()

    def get_channel(self, name: str) -> Channel:
        for channel in self:
            if channel.name == name:
                return channel
        return self.add_channel(name)

    def add_channel(self, name: str) -> Channel:
        channel: Channel = Channel(name)
        self.append(channel)
        return channel
