from typing_extensions import Self
from torch import Tensor
import torch


class TensorBuilder:
    tensor: Tensor = torch.empty(size=(1, 0), dtype=torch.int64)

    def build(self) -> Tensor:
        return self.tensor

    def append(self, tensor: Tensor) -> Self:
        self.tensor = torch.cat((self.tensor, tensor), dim=1)
        return self

    def appendNTimes(self, tensor: Tensor, n: int) -> Self:
        for _ in range(n):
            self.append(tensor)
        return self

    def tail(self, n: int) -> Self:
        self.tensor = self.tensor[-n:]
        return self

    def removeInitialTokens(self, n: int) -> Self:
        self.tensor = self.tensor[:, n:]
        return self
