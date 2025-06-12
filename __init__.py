import os

QUADDIF_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
QUADDIF_ENVS_DIR = os.path.join(QUADDIF_ROOT_DIR, 'envs')

print("The root dir of quaddif:", QUADDIF_ROOT_DIR)

import env
import algo
import network
import script
import utils

__all__ = [
    "env",
    "algo",
    "network",
    "script",
    "utils",
]