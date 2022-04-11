import numpy as np

from Rungekutta.Model.Term.Term import Term


class D3Dx(Term):

    def _get_runge_kutta_value(self, dt, dx, u: np.ndarray) -> np.ndarray:
        return self.value * (np.append(u[2:], [u[1], u[2]]) - 3 * np.append(u[1:], u[1]) + 3 * u - np.append(u[-2], u[
                                                                                                                    :-1])) * dt / dx ** 3
