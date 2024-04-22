# Adapted from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/tree/master/code/dataset/__init__.py
from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from . import utils

_type = {"cars": Cars, "cub": CUBirds, "SOP": SOP}


def load(name, root, mode, transform=None):
    return _type[name](root=root, mode=mode, transform=transform)
