import glob

import numpy as np


class GetEPICData:
    __step = 1
    __x_range = [400, 600]
    __time_range = [400, 480]

    @staticmethod
    def get_data(path: str) -> np.ndarray:
        data_files = sorted(glob.glob(path + '/*'))
        data_list = []
        for i in range(GetEPICData.__time_range[0], GetEPICData.__time_range[1], GetEPICData.__step):
            filename = data_files[i]
            temp = np.loadtxt(filename)
            data_list.append(temp[GetEPICData.__x_range[0]:GetEPICData.__x_range[1], :])
        return np.array(data_list)
