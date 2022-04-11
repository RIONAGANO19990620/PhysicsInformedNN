from Rungekutta.ForNN.Util import Util
from Rungekutta.Model.Term.D1Dx import D1Dx
from Rungekutta.Model.Term.D2Dx import D2Dx
from Rungekutta.Model.Term.D3Dx import D3Dx


def get_advection():
    term_list = [D1Dx(- Util.a)]
    data = Util.get_data(term_list)
    teacher_data = Util.get_teacher_data(term_list)
    Util.plot_teacher_data(teacher_data)
    return teacher_data, data


def get_advection_diffusion():
    term_list = [D1Dx(- Util.a), D2Dx(Util.b)]
    data = Util.get_data(term_list)
    teacher_data = Util.get_teacher_data(term_list)
    Util.plot_teacher_data(teacher_data)
    return teacher_data, data


def get_burgers():
    term_list = [D1Dx(Util.a, True), D2Dx(Util.b)]
    data = Util.get_data(term_list)
    teacher_data = Util.get_teacher_data(term_list)
    Util.plot_teacher_data(teacher_data)
    return teacher_data, data


def get_kdv():
    term_list = [D1Dx(-Util.a, True), D3Dx(-Util.b)]
    data = Util.get_data(term_list)
    teacher_data = Util.get_teacher_data(term_list)
    Util.plot_teacher_data(teacher_data)
    return teacher_data, data
