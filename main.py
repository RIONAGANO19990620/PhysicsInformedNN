from GetRungekuttaData import Util, GetRungekuttaData
from PhysicsInformedNN import PhysicsInformedNN

x_array = Util.mesh.x_array
t_array = Util.mesh.t_array


def pred_advection(train_num=1000):
    teacher_data, data = GetRungekuttaData.get_advection()
    advection = PhysicsInformedNN(x_array, t_array, teacher_data)
    advection.train(train_num)
    advection.print_coeffisient()
    advection.save_plot_u(data, 'advection')
    advection.save_print_coeffisient('advection')


def pred_kdv(train_num=10000):
    teacher_data, data = GetRungekuttaData.get_kdv()
    kdv = PhysicsInformedNN(x_array, t_array, teacher_data)
    kdv.train(train_num)
    kdv.print_coeffisient()
    kdv.save_plot_u(data, 'kdv')
    kdv.save_print_coeffisient('kdv')

if __name__ == '__main__':
    pred_kdv()
