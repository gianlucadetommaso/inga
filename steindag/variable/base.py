from torch import Tensor
from abc import abstractmethod
from typing import Iterable


class Variable:
    def __init__(self, name: str, parent_names: Iterable[str], sigma: float) -> None:
        self.name = name
        self.parent_names = parent_names
        self.sigma = sigma

    def f(
        self, parents: dict[str, Tensor], u: Tensor, f_bar: Tensor | None = None
    ) -> Tensor:
        if f_bar is None:
            f_bar = self.f_bar(parents)
        return f_bar + self.sigma * u

    @abstractmethod
    def f_bar(self, parents: dict[str, Tensor]) -> Tensor: ...
