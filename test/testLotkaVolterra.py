import unittest

import numpy as np
from matplotlib import pyplot as plt

from GetRungekuttaData import Util


class TestLotkaVolterra(unittest.TestCase):

    def setUp(self) -> None:
        self.t = Util.mesh.t_array
        self.n = self.t.shape[0]
        self.h = self.t / self.n

    def test_getlotka_volterra(self):
        # 区間の分割の設定
        T = 50
        n = 100000
        h = T / n
        t = np.arange(0, T, h)

        # 方程式を定める関数、初期値の定義
        f = lambda u, v, t=0: u - u * v
        g = lambda u, v, t=0: u * v - v
        u_0 = 2
        v_0 = 1.1

        # 結果を返すための配列の宣言
        u = np.empty(n)
        v = np.empty(n)
        u[0] = u_0
        v[0] = v_0

        # 方程式を解くための反復計算
        for i in range(n - 1):
            k_1 = h * f(u[i], v[i], t[i])
            k_2 = h * f(u[i] + k_1 / 2, v[i], t[i] + h / 2)
            k_3 = h * f(u[i] + k_2 / 2, v[i], t[i] + h / 2)
            k_4 = h * f(u[i] + k_3, v[i], t[i] + h)
            u[i + 1] = u[i] + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            j_1 = h * g(u[i], v[i], t[i])
            j_2 = h * g(u[i], v[i] + j_1 / 2, t[i] + h / 2)
            j_3 = h * g(u[i], v[i] + j_2 / 2, t[i] + h / 2)
            j_4 = h * g(u[i], v[i] + j_3, t[i] + h)
            v[i + 1] = v[i] + 1 / 6 * (j_1 + 2 * j_2 + 2 * j_3 + j_4)

        # グラフで可視化
        plt.plot(t, u, label="u")
        plt.plot(t, v, label="v")
        plt.show()
