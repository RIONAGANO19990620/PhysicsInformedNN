import numpy as np

from Rungekutta.Model.Term.Term import Term


class D1Dx(Term):

    def _get_runge_kutta_value(self, dt, dx, u: np.ndarray) -> np.ndarray:
        return self.value * ((2 * np.append(u[1:], u[1]) + 3 * u - 6 * np.append(u[-2],
                                                                    u[:- 1]) + np.append(
            [u[-3], u[-2]], u[
                            :-2])) * dt / 6 / dx)
