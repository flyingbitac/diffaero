import json
import os
import numpy as np

def get_idle_device(n: int = 1):
    gpu_info = json.loads(os.popen("gpustat --json").read())
    mems = np.array([gpu["memory.used"] / gpu["memory.total"] for gpu in gpu_info["gpus"]]) + 0.01
    utils = np.array([gpu["utilization.gpu"] * 0.01 for gpu in gpu_info["gpus"]]) + 0.01
    powers = np.array([gpu["power.draw"] / gpu["enforced.power.limit"] for gpu in gpu_info["gpus"]]) + 0.01
    usage = utils * powers
    if n == 1:
        return int(usage.argmin())
    else:
        return tuple(map(lambda x: int(x), usage.argsort()[:n]))

if __name__ == "__main__":
    print(get_idle_device())
    print(get_idle_device(2))