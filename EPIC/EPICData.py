from typing import List

import numpy as np

from EPIC.GetEPICData import GetEPICData


class EPICData:
    def __init__(self, path):
        self.data_list = GetEPICData.get_data(path)

    @property
    def x_array(self) -> np.ndarray:
        return self.data_list[0][:, 0]

    @property
    def t_array(self) -> np.ndarray:
        return np.linspace(0, self.data_list.shape[0] - 1, self.data_list.shape[0])

    @property
    def data(self) -> List[np.ndarray]:
        data = []
        for i in range(len(self.data_list)):
            data.append(self.data_list[i][:, 1])
        return data

    @property
    def teacher_data(self) -> np.ndarray:
        teacher_data = np.array(self.data).flatten()[:, None]
        return (teacher_data - teacher_data.min()) / (teacher_data.max() - teacher_data.min())
