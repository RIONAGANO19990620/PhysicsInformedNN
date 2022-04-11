from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np


class Term(metaclass=ABCMeta):

    def __init__(self, value: float, variable=False):
        self.value = value
        self.variable = variable

    def get_runge_kutta_value(self, dt, dx, u: np.ndarray) -> np.ndarray:
        if self.variable:
            return u * self._get_runge_kutta_value(dt, dx, u)
        return self._get_runge_kutta_value(dt, dx, u)

    @abstractmethod
    def _get_runge_kutta_value(self, dt, dx, u: np.ndarray) -> np.ndarray:
        pass
