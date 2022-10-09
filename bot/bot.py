from argparse import Namespace
from twitchio.ext import commands
from util.MessageHandler import MessageHandler, Pipeline
from util.Channel import ChannelList
from model.GPT2Model import GPT2Model
from model.GPT2Tokenizer import GPT2Tokenizer
from util.EnvDefault import ArgParser
from util.CustomLogger import CustomLogger
from util.TwitchAuth import TwitchAuth
from util.CustomLogger import CustomLogger
import logging

log: logging.Logger = CustomLogger(__name__).get_logger()


class Bot(commands.Bot):
    args: Namespace = ArgParser().get_args_from_env()
    handler: MessageHandler
    auth: TwitchAuth = TwitchAuth(
        client_id=args.client_id,
        client_secret=args.client_secret,
    )

    def __init__(self, handler: MessageHandler) -> None:
        self.handler = handler
        super().__init__(
            client_secret=self.args.client_secret,
            client_id=self.args.client_id,
            nick=self.args.nick,
            initial_channels=self.args.initial_channels.split(","),
            token=self.auth.get_token(),
            prefix="!",
        )

    async def event_ready(self) -> None:
        log.info(f"Logged in as {self.nick}")

    async def event_message(self, message) -> None:
        await handler.handle_message(message)


if __name__ == "__main__":
    handler: MessageHandler = MessageHandler(GPT2Model(), GPT2Tokenizer())
    bot: Bot = Bot(handler)
    bot.run()
