import re
import os


def remove_self_mentions(text) -> str:
    pattern: re.Pattern = re.compile(get_nick(), re.IGNORECASE)
    return re.sub(pattern, "", text)


def contains_self_mention(text) -> bool:
    pattern: re.Pattern = re.compile(get_nick(), re.IGNORECASE)
    return re.search(pattern, text) is not None


def get_nick() -> str:
    nick: str | None = os.getenv("NICK")
    assert nick, "NICK environment variable not set"
    return nick
