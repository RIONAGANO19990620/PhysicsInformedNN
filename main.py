from EPIC.EPICData import EPICData
from Execution.GetRungekuttaData import Util, GetRungekuttaData
from Execution.PathManager import PathManager
from Execution.PhysicsInformedNN import PhysicsInformedNN

x_array = Util.mesh.x_array
t_array = Util.mesh.t_array


def pred_advection(train_num=100):
    teacher_data, data = GetRungekuttaData.get_advection()
    advection = PhysicsInformedNN(x_array, t_array, teacher_data)
    path = PathManager.get_advection()
    advection.train(train_num)
    advection.print_coeffisient(path)
    advection.save_plot_u(data, 'advection', path)
    advection.save_plot_coeffisient('advection', path)


def pred_kdv(train_num=100):
    teacher_data, data = GetRungekuttaData.get_kdv()
    kdv = PhysicsInformedNN(x_array, t_array, teacher_data)
    kdv.train(train_num)
    path = PathManager.get_kdv()
    kdv.print_coeffisient(path)
    kdv.save_plot_gif(path)
    kdv.save_plot_coeffisient('kdv', path)


def pred_burgers(train_num=100):
    teacher_data, data = GetRungekuttaData.get_burgers()
    burgers = PhysicsInformedNN(x_array, t_array, teacher_data)
    burgers.train(train_num)
    path = PathManager.get_burgers()
    burgers.print_coeffisient(path)
    burgers.save_plot_u(data, 'burgers', path)
    burgers.save_plot_coeffisient('burgers', path)


def pred_advection_diffusion(train_num=100):
    teacher_data, data = GetRungekuttaData.get_advection_diffusion()
    advection_diffusion = PhysicsInformedNN(x_array, t_array, teacher_data)
    advection_diffusion.train(train_num)
    path = PathManager.get_advection_diffusion()
    advection_diffusion.print_coeffisient(path)
    advection_diffusion.save_plot_u(data, 'advection_diffusion', path)
    advection_diffusion.save_plot_coeffisient('advection_diffusion', path)


def pred_epic_data(train_num=1000):
    path = './EPIC/Ion_Density_y'
    epic_data = EPICData(path)
    epic = PhysicsInformedNN(epic_data.x_array, epic_data.t_array, epic_data.teacher_data)
    path = PathManager.get_epic()
    epic.train(train_num)
    epic.print_coeffisient(path)
    epic.save_plot_gif(path)
    epic.save_plot_coeffisient('epic', path)


if __name__ == '__main__':
    pred_epic_data(100)
