import unittest

from GetRungekuttaData import Util, GetRungekuttaData
from PhysicsInformedNN import PhysicsInformedNN


class TestPINNs(unittest.TestCase):

    def setUp(self) -> None:
        mesh = Util.mesh
        self.x_array = mesh.x_array
        self.t_array = mesh.t_array

    def test_advection(self):
        teacher_data, data = GetRungekuttaData.get_advection()
        advection = PhysicsInformedNN(self.x_array, self.t_array, teacher_data)
        advection.train()
