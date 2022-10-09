import torch
from torch import Tensor

from model.LanguageModel import LanguageModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as BaseModel
from util.CustomLogger import CustomLogger
from logging import Logger
import config.Config as Config
from transformers import StoppingCriteria

from util.TensorBuilder import TensorBuilder

log: Logger = CustomLogger(__name__).get_logger()


class GPT2Model(LanguageModel):
    wrappedModel: BaseModel
    device: torch.device

    def __init__(self) -> None:
        super().__init__()
        self.device = self.set_device_from_config()
        self.wrappedModel = BaseModel.from_pretrained(Config.MODEL_PATH).to(
            Config.DEVICE
        )

    def generate(self, tokens: Tensor) -> Tensor:
        tokens: Tensor = self.trim_tokens_to_max_size(tokens)
        log.debug(f"generate({tokens})")
        attentionMask: Tensor = torch.ones_like(tokens)

        generated: Tensor = self.wrappedModel.generate(
            tokens.to(Config.DEVICE),
            attention_mask=attentionMask.to(Config.DEVICE),
            max_length=self.get_max_length(),
            temperature=Config.TEMPERATURE,
            top_k=Config.TOP_K,
            top_p=Config.TOP_P,
            no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=Config.REPETITION_PENALTY,
            min_length=Config.MIN_LENGTH,
            eos_token_id=Config.NEWLINE_TOKEN_ID,
            pad_token_id=Config.NEWLINE_TOKEN_ID,
            do_sample=True,
            num_return_sequences=1,
        )[0]

        # strip input tokens from output
        return generated[tokens.shape[1] :]

    def trim_tokens_to_max_size(self, tokens: Tensor) -> Tensor:
        return tokens[-self.get_max_length() :]

    def get_max_length(self) -> int:
        return self.wrappedModel.config.n_positions

    def set_device_from_config(self) -> torch.device:
        if (Config.DEVICE == "cuda:0") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
