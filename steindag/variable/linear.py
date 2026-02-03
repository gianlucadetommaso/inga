from torch import Tensor
from typing import Iterable
from steindag.variable.base import Variable
    

class LinearVariable(Variable):
    def __init__(self, name: str, parent_names: Iterable[str], sigma: float, coefs: dict[str, float], intercept: float) -> None:
        super().__init__(name, parent_names, sigma)
        
        self._coefs = coefs
        self._intercept = intercept
        
    def f_bar(self, parents: dict[str, Tensor]) -> Tensor:
        f_bar = self._intercept
        for parent_name, parent in parents.items():
            f_bar += self._coefs[parent_name] * parent
            
        return f_bar