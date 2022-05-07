from typing import List

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from EPIC.EPICData import EPICData


class PlotEPICData:

    @staticmethod
    def plot_gif(data: List[np.ndarray], save_path):
        fig = plt.figure()
        ims = []

        for simple_data in data:
            im = plt.plot(simple_data, color='red')
            plt.xlabel('x')
            ims.append(im)

        ani = animation.ArtistAnimation(fig, ims, interval=1)

        ani.save(save_path, writer='imagemagick')


if __name__ == '__main__':
    path = 'Ion_Density_y'
    data = EPICData(path)
    PlotEPICData.plot_gif(data.data, 'sample.gif')
