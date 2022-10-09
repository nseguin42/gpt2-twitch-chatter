import argparse
import os


class EnvDefault(argparse.Action):
    "Allow argparse to use environment variables as defaults."

    def __init__(self, envvar, required=True, default=None, **kwargs) -> None:
        default_or_env: str | None = self.get_default_else_env(envvar, default)
        is_still_required: bool = required and not default_or_env
        super(EnvDefault, self).__init__(
            default=default_or_env, required=is_still_required, **kwargs
        )

    def get_default_else_env(self, envvar, default) -> str | None:
        if default:
            return default

        if envvar in os.environ:
            return os.environ[envvar]

    def is_still_required(self, required, default_or_env) -> bool:
        return required and not default_or_env

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)


class ArgParser(argparse.ArgumentParser):
    "Set defaults from environment variables."

    def add_argument(self, *args, **kwargs) -> None:
        if "envvar" in kwargs:
            kwargs["action"] = EnvDefault
        super().add_argument(*args, **kwargs)

    def get_args_from_env(self) -> argparse.Namespace:
        arg_parser: ArgParser = ArgParser()
        arg_parser.add_argument("--nick", envvar="NICK")
        arg_parser.add_argument("--client_id", envvar="CLIENT_ID")
        arg_parser.add_argument("--client_secret", envvar="CLIENT_SECRET")
        arg_parser.add_argument("--initial_channels", envvar="CHANNELS")
        return arg_parser.parse_args()
