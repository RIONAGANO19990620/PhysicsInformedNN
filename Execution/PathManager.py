import os
import pathlib


class PathManager:
    __root_dir_name = 'PhysicsInformedNN'

    @staticmethod
    def get_root_path():
        return PathManager.__get_root_recursive(pathlib.Path(__file__))

    @staticmethod
    def __get_root_recursive(path: pathlib.Path):
        dir_path = os.path.dirname(path)
        if os.path.basename(dir_path) == PathManager.__root_dir_name:
            return pathlib.Path(dir_path)
        return PathManager.__get_root_recursive(dir_path)

    @staticmethod
    def get_data_dir():
        return PathManager.get_root_path() / 'Data'

    @staticmethod
    def get_advection():
        return PathManager.get_data_dir() / 'Advection'

    @staticmethod
    def get_advection_diffusion():
        return PathManager.get_data_dir() / 'AdvectionDiffusion'

    @staticmethod
    def get_burgers():
        return PathManager.get_data_dir() / 'Burgers'

    @staticmethod
    def get_kdv():
        return PathManager.get_data_dir() / 'Kdv'

    @staticmethod
    def get_epic():
        return PathManager.get_data_dir() / 'EPIC'
