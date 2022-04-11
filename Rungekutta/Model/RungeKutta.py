from typing import Iterator, List

import numpy as np
from matplotlib import pyplot as plt

from Rungekutta.Model.Term.Term import Term


class RungeKutta:

    @staticmethod
    def solve(term_list: List[Term], first_data: np.ndarray, T_mesh: int, T_change: int, dt, dx):
        second_data = first_data + RungeKutta.__calculate_equation(term_list, first_data, dt, dx)
        data_list = [first_data, second_data]

        for _ in range(T_mesh - 2):
            data_list = RungeKutta.__solve_unit_time(term_list, data_list, dt, dx)

        data = [data_list[n * T_change] for n in range(T_mesh // T_change)]

        return data

    @staticmethod
    def plot(data, t_array, x_array, title: str):
        for t_n in range(len(t_array)):
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('u')
            if t_n % (len(t_array) // 5) == 0 or t_n == len(t_array) - 1: plt.plot(x_array, data[t_n],
                                                                               label=f"t={str(t_array[t_n])[: 4]}")

    @staticmethod
    def __solve_unit_time(term_list: List[Term], datalist: List[np.ndarray], dt, dx) -> List[np.ndarray]:
        u_one_back = datalist[-1]
        u_two_back = datalist[-2]

        k1 = RungeKutta.__calculate_equation(term_list, u_one_back, dt, dx)
        u_one_asterisk = u_two_back + k1 / 2
        k2 = RungeKutta.__calculate_equation(term_list, u_one_asterisk, dt, dx)
        u_two_asterisk = u_two_back + k2 / 2
        k3 = RungeKutta.__calculate_equation(term_list, u_two_asterisk, dt, dx)
        u_three_asterisk = u_two_back + k3
        k4 = RungeKutta.__calculate_equation(term_list, u_three_asterisk, dt, dx)
        next_u = u_one_back + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        datalist.append(next_u)
        return datalist

    @staticmethod
    def __calculate_equation(term_list: List[Term], u: np.ndarray, dt, dx) -> np.ndarray:
        k = np.zeros(np.shape(u))
        for term in term_list:
            k += term.get_runge_kutta_value(dt, dx, u)
        return k
