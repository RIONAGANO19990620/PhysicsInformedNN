import dataclasses

import numpy as np


@dataclasses.dataclass
class NormalizedData:
    x_array: np.ndarray
    t_array: np.ndarray
    u_array: np.ndarray

    @property
    def u_max(self):
        return self.u_array.max()

    @property
    def u_min(self):
        return self.u_array.min()

    @property
    def x_max(self):
        return self.x_array.max()

    @property
    def x_min(self):
        return self.x_array.min()

    @property
    def t_max(self):
        return self.t_array.max()

    @property
    def t_min(self):
        return self.t_array.min()

    @property
    def normalized_u(self) -> np.ndarray:
        return (self.u_array - self.u_min) / (self.u_max - self.u_min)

    @property
    def normalized_t(self) -> np.ndarray:
        return (self.t_array - self.t_min) / (self.t_max - self.t_min)

    @property
    def normalized_x(self) -> np.ndarray:
        return (self.x_array - self.x_min) / (self.x_max - self.x_min)