from twitchio import Message


class MessageIgnorer:
    message: Message
    bot_usernames: list[str] = [
        "nightbot",
        "moobot",
        "streamlabs",
        "streamelements",
        "autoraver",
        "poopthefirst",
        "markov_chain_bot",
        "sumbot",
        "scripbozo",
    ]
    shouldIgnore: bool = False

    def __init__(self, message: Message) -> None:
        self.message = message

    def should_ignore(self) -> bool:
        return self.should_always_ignore() or (
            self.should_usually_ignore() and not self.should_never_ignore()
        )

    def should_always_ignore(self) -> bool:
        return self.is_own_message()

    def should_usually_ignore(self) -> bool:
        return self.is_user_bot() or self.is_message_empty()

    def is_message_empty(self) -> bool:
        return not self.message.content

    def should_never_ignore(self) -> bool:
        return self.is_user_privileged()

    def is_own_message(self) -> bool:
        return self.message.echo

    def is_user_bot(self) -> bool:
        return self.message.author.name in self.bot_usernames

    def is_user_privileged(self) -> bool:
        return self.is_user_mod() or self.is_user_broadcaster()

    def is_user_mod(self) -> bool:
        return self.message.tags["mod"]

    def is_user_broadcaster(self) -> bool:
        return self.message.tags["broadcaster"]

    def is_user_subscriber(self) -> bool:
        return self.message.tags["subscriber"]
