from typing import List

import numpy as np
from matplotlib import pyplot as plt

from Rungekutta.Model.Mesh import Mesh
from Rungekutta.Model.RungeKutta import RungeKutta
from Rungekutta.Model.Term.D1Dx import D1Dx
from Rungekutta.Model.Term.D2Dx import D2Dx
from Rungekutta.Model.Term.D3Dx import D3Dx
from Rungekutta.Model.Term.Term import Term


class Util:
    X_max = 10
    X_mesh = 100
    T_max = 8
    T_change = 200
    mesh = Mesh(X_max, X_mesh, T_max, T_change)
    a = 1
    b = 0.1
    first_data = np.exp(-(mesh.x_array - mesh.X_max // 2) ** 2 / 4)

    @staticmethod
    def get_data(term_list: List[Term]):
        data = RungeKutta.solve(term_list, Util.first_data, Util.mesh.T_mesh, Util.mesh.T_change, Util.mesh.dt,
                                Util.mesh.dx)
        return data

    @staticmethod
    def get_teacher_data(term_list: List[Term]):
        data = Util.get_data(term_list)
        teacher_data = np.array(data).flatten()[:, None]
        noise = 0.1
        teacher_data_noisy = teacher_data + noise * np.std(teacher_data) * np.random.randn(teacher_data.shape[0],
                                                                                           teacher_data.shape[1])
        return teacher_data_noisy

    @staticmethod
    def plot_teacher_data(teacher_data):
        teacher_data_reshaped = teacher_data.reshape(len(Util.mesh.t_array), len(Util.mesh.x_array))
        for t_n in range(len(Util.mesh.t_array)):
            plt.title('advection equation noisy')
            plt.xlabel('x')
            plt.ylabel('u')
            if t_n % (len(Util.mesh.t_array) // 5) == 0 or t_n == len(Util.mesh.t_array) - 1:
                plt.plot(Util.mesh.x_array, teacher_data_reshaped[t_n], label=f" t={str(Util.mesh.t_array[t_n])[:4]}")
        plt.legend()
        plt.show()


class GetRungekuttaData:

    @staticmethod
    def get_advection():
        term_list = [D1Dx(- Util.a)]
        data = Util.get_data(term_list)
        teacher_data = Util.get_teacher_data(term_list)
#        Util.plot_teacher_data(teacher_data)
        return teacher_data, data

    @staticmethod
    def get_advection_diffusion():
        term_list = [D1Dx(- Util.a), D2Dx(Util.b)]
        data = Util.get_data(term_list)
        teacher_data = Util.get_teacher_data(term_list)
 #       Util.plot_teacher_data(teacher_data)
        return teacher_data, data

    @staticmethod
    def get_burgers():
        term_list = [D1Dx(Util.a, True), D2Dx(Util.b)]
        data = Util.get_data(term_list)
        teacher_data = Util.get_teacher_data(term_list)
#        Util.plot_teacher_data(teacher_data)
        return teacher_data, data

    @staticmethod
    def get_kdv():
        term_list = [D1Dx(-Util.a, True), D3Dx(-Util.b)]
        data = Util.get_data(term_list)
        teacher_data = Util.get_teacher_data(term_list)
#        Util.plot_teacher_data(teacher_data)
        return teacher_data, data
