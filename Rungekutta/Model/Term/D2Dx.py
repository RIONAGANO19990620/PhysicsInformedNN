import numpy as np

from Rungekutta.Model.Term.Term import Term


class D2Dx(Term):

    def _get_runge_kutta_value(self, dt, dx, u: np.ndarray) -> np.ndarray:
        return self.value * ((
                                     - np.append(u[2:], [u[1], u[2]]) + 16 * np.append(u[1:],
                                                                                       u[1]) - 30 * u + 16 * np.append(
                                 u[-2], u[:-
                                 1]) - np.append([u[-3], u[-2]], u[:-2])) * dt / dx ** 2 / 12)
