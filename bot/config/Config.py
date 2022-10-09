DEFAULT_LOG_LEVEL: str = "INFO"
ERROR_MESSAGE_COULD_NOT_GENERATE: str = "I don't know what to say!"

TWITCH_AUTH_JSON: str = "twitch_auth.json"


# Model
MODEL_PATH: str = "/home/kogasa/git/ns/gpt2-twitch-chatter/data/output/model"
DEVICE: str = "cuda:0"
NEWLINE_TOKEN_ID: int = 198  # GPT2Tokenizer.from_pretrained("gpt2").encode("\n")[0]
MODEL_MAX_LENGTH: int = 512
OUTPUT_MAX_LENGTH: int = 64


# Generation
TEMPERATURE: float = 0.6
TOP_K: int = 0
TOP_P: float = 0.92
NO_REPEAT_NGRAM_SIZE: int = 6
REPETITION_PENALTY: float = 1.6
PROMPT_DUPLICATION_FACTOR: int = 3
MIN_LENGTH: int = 1

RATE_LIMIT_SECONDS: int = 30
CLIENT_TIMEOUT: int = 10
