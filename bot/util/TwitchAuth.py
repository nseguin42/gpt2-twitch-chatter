from dataclasses import dataclass, field
from math import ceil
from typing import Any
from typing_extensions import Self
import ujson
import requests
from datetime import datetime
import logging
import config.Config as Config
from util.CustomLogger import CustomLogger

TWITCH_OAUTH_URL: str = "https://id.twitch.tv/oauth2/token"


log: logging.Logger = CustomLogger(__name__).get_logger()


@dataclass
class TwitchAuthData:
    "Data type for the response from the Twitch API."
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        return ujson.dumps(self.data)

    def write_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            ujson.dump(self.data, f, indent=4)

    def from_json(self, data: str) -> Self:
        self.data = ujson.loads(data)
        return self

    def get_auth_token(self) -> str:
        return self.data["auth_token"]

    def get_expiration_time(self) -> int:
        return self.data["creation_time"] + self.data["expires_in"]

    def get_refresh_token(self) -> str:
        return self.data["refresh_token"]

    def from_response(self, response: requests.Response) -> Self:
        self.data["auth_token"] = response.json()["access_token"]
        self.data["expires_in"] = response.json()["expires_in"]
        self.data["refresh_token"] = response.json()["refresh_token"]
        self.data["creation_time"] = ceil(datetime.now().timestamp())
        return self

    def is_expired(self) -> bool:
        log.debug(
            f"Refresh token expires at {datetime.fromtimestamp(self.get_expiration_time())}."
        )
        return self.get_expiration_time() < ceil(datetime.now().timestamp())


class TwitchAuth:
    client_id: str
    client_secret: str
    data: TwitchAuthData = TwitchAuthData()

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.setup_refresh_token()

    def setup_refresh_token(self) -> None:
        try:
            self.data = self.load_from_file()
        except Exception as e:
            log.exception(e)
            self.data = TwitchAuthData()

        if self.data.is_expired():
            self.refresh_token()

    def load_from_file(self) -> TwitchAuthData:
        log.info(f"Loading auth data from {Config.TWITCH_AUTH_JSON}")
        with open(Config.TWITCH_AUTH_JSON, "r") as f:
            file: str = f.read()
            data: TwitchAuthData = TwitchAuthData().from_json(file)
            if not data:
                raise Exception("No Twitch auth data found in file.")
            return data

    def save_to_file(self) -> None:
        log.info(f"Saving new auth data to {Config.TWITCH_AUTH_JSON}")
        with open(Config.TWITCH_AUTH_JSON, "w") as f:
            f.write(self.data.to_json())

    def request_new_token_using_refresh(self) -> Any:
        log.info(f"Requesting new token using refresh token.")
        url: str = "https://id.twitch.tv/oauth2/token"
        params: dict[str, str] = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.data.get_refresh_token(),
        }
        response: requests.Response = requests.post(url, params=params)
        return TwitchAuthData().from_response(response)

    def update_data(self, data: TwitchAuthData) -> None:
        self.data = data
        self.save_to_file()

    def refresh_token(self) -> None:
        log.info("Refreshing token.")
        try:
            data: TwitchAuthData = self.request_new_token_using_refresh()
            self.update_data(data)
        except Exception as e:
            log.exception(e)

    def get_token(self) -> str:
        if not self.data or self.data.is_expired():
            self.refresh_token()
        return self.data.get_auth_token()
