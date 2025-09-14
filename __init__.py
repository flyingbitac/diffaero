import os

DIFFAERO_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DIFFAERO_ENVS_DIR = os.path.join(DIFFAERO_ROOT_DIR, 'envs')

print("The root dir of DiffAero:", DIFFAERO_ROOT_DIR)

from . import env
from . import algo
from . import network
from . import script
from . import utils

__all__ = [
    "env",
    "algo",
    "network",
    "script",
    "utils",
]