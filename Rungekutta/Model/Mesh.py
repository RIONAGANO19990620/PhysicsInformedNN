import dataclasses

import numpy as np


@dataclasses.dataclass
class Mesh:
    X_max: float
    X_mesh: int
    T_max: float
    T_change: int

    @property
    def dx(self) -> float:
        return self.X_max / self.X_mesh

    @property
    def T_mesh(self) -> int:
        return 100 * self.T_change

    @property
    def dt(self) -> float:
        return self.T_max / self.T_mesh

    @property
    def x_array(self) -> np.ndarray:
        return np.linspace(0, self.X_max, self.X_mesh)

    @property
    def t_array(self) -> np.ndarray:
        return np.linspace(0, self.T_max, self.T_mesh // self.T_change)