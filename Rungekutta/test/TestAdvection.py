import unittest

import numpy as np
from matplotlib import pyplot as plt

from Rungekutta.Model.RungeKutta import RungeKutta
from Rungekutta.Model.Term.D1Dx import D1Dx
from Rungekutta.Model.Term.D2Dx import D2Dx
from Rungekutta.Model.Term.D3Dx import D3Dx


class TestAdvection(unittest.TestCase):
    X_max = 10
    X_mesh = 100
    T_max = 8
    T_change = 200  # 時間方向のメッシュ数を増加するために任意の値を使用 T_mesh = 100 * T_change
    dx = X_max / X_mesh
    T_mesh = 100 * T_change
    dt = T_max / T_mesh
    x_array = np.linspace(0, X_max, X_mesh)
    t_array = np.linspace(0, T_max, T_mesh // T_change)

    def test_advection(self):
        a = 1
        b = 0.1
        term = D1Dx(-a)
        first_data = np.exp(-(self.x_array - self.X_max // 2) ** 2 / 4)
        term_list = [term]
        data = RungeKutta.solve(term_list, first_data, self.T_mesh, self.T_change, self.dt, self.dx)
        RungeKutta.plot(data, self.t_array, self.x_array, 'advection')
        plt.legend()
        plt.show()

    def test_advection_diffusion(self):
        a = 1
        b = 0.1
        term_list = [D1Dx(-a), D2Dx(b)]
        first_data = np.exp(-(self.x_array - self.X_max // 2) ** 2 / 4)
        data = RungeKutta.solve(term_list, first_data, self.T_mesh, self.T_change, self.dt, self.dx)
        RungeKutta.plot(data, self.t_array, self.x_array, 'advection_diffusion')
        plt.legend()
        plt.show()

    def test_burgers(self):
        a = 1
        b = 0.1
        first_data = np.exp(-(self.x_array - self.X_max // 2) ** 2 / 4)
        term_list = [D1Dx(-a, True), D2Dx(b)]
        data = RungeKutta.solve(term_list, first_data, self.T_mesh, self.T_change, self.dt, self.dx)
        RungeKutta.plot(data, self.t_array, self.x_array, 'burgers')
        plt.legend()
        plt.show()

    def test_kdv(self):
        a = 1
        b = 0.1
        first_data = np.exp(-(self.x_array - self.X_max // 2) ** 2 /4)
        term_list = [D1Dx(-a, True), D3Dx(-b)]
        data = RungeKutta.solve(term_list, first_data, self.T_mesh, self.T_change, self.dt, self.dx)
        RungeKutta.plot(data, self.t_array, self.x_array, 'kdv')
        plt.legend()
        plt.show()
