# DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning

This repository contains the code of the paper: [DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning](https://arxiv.org/abs/2509.10247)

- [DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning](#diffaero-a-gpu-accelerated-differentiable-simulation-framework-for-efficient-quadrotor-policy-learning)
  - [Introduction](#introduction)
  - [Features](#features)
    - [Environments](#environments)
    - [Learning algorithms](#learning-algorithms)
    - [Dynamical models](#dynamical-models)
    - [Sensors](#sensors)
  - [Installation](#installation)
    - [System requirements](#system-requirements)
    - [Installing the DiffAero](#installing-the-diffaero)
  - [Usage](#usage)
    - [Basic usage](#basic-usage)
    - [Visualization](#visualization)
      - [Visualization with taichi GUI](#visualization-with-taichi-gui)
      - [Visualize the depth camera and LiDAR data](#visualize-the-depth-camera-and-lidar-data)
      - [Record First-Person View Videos](#record-first-person-view-videos)
    - [Sweep across multiple configurations](#sweep-across-multiple-configurations)
      - [Sweep across multiple GPUs in parallel](#sweep-across-multiple-gpus-in-parallel)
      - [Automatic Hyperparameter Tuning](#automatic-hyperparameter-tuning)
  - [Deploy](#deploy)
  - [TODO-List](#todo-list)
  - [Citation](#citation)

## Introduction

DiffAero is a GPU-accelerated differentiable quadrotor simulator that parallelizes both physics and rendering. It achieves orders-of-magnitude performance improvements over existing platforms with little VRAM consumption. It provides a modular and extensible framework supporting four differentiable dynamics models, three sensor modalities, and three flight tasks. Its PyTorch-based interface unifies four learning formulations and three learning paradigms. This flexibility enables DiffAero to serve as a benchmark for learning algorithms and allows researchers to investigate a wide range of problems, from differentiable policy learning to multi-agent coordination. Users can combine different components almost arbitrarily to initiate a custom-configured training process with minimal effort.

## Features
<!-- Inserted English summary table -->
| Module         | Currently Supported                                                     |
|----------------|-------------------------------------------------------------------------|
| Tasks          | Position Control, Obstacle Avoidance, Racing                            |
| Differential  Learning Algorithms     | BPTT, SHAC, SHA2C                                |
| Reinforcement Learning Algorithms     | PPO, Dreamer V3                                  |
| Sensors        | Depth Camera, LiDAR                                                     |
| Dynamic Models | Full Quadrotor, Continuous Point-Mass, Discrete Point-Mass                        |

### Environments

DiffAero now supports three flight tasks: 
- **Position Control** (`env=pc`): The goal is to navigate to and hover on the specified target positions from random initial positions, without colliding with other agents.
- **Obstacle Avoidance** (`env=oa`): The goal is to navigate to and hover on target positions while avoiding collision with environmental obstacles and other quadrotors, given exteroceptive informations:
  - Relative positions of obstacles w.r.t. the quadrotor, or
  - Image from the depth camera attached to the quadrotor, or
  - Ray distance from the LiDAR attached to the quadrotor.
- **Racing** (`env=racing`): The goal is to navigate through a series of gates in the shortest time, without colliding with the gates.

### Learning algorithms

We have implemented several learning algorithms, including RL algorithms and algorithms that exploit the differentiability of the simulator:

- **Reinforcement Learning algorithms**:
    - **PPO** (`algo=ppo`): [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
    - **Dreamer V3** (`algo=world`): [Mastering Diverse Domains through World Models](http://arxiv.org/abs/2301.04104)

- **Differential algorithms**:
    - **BPTT** (`algo=apg(_sto)`): Direct back-propagation through time, supports deterministic policy (`algo=apg`) and stochastic policy (`algo=apg_sto`)
    - **SHAC** (`algo=shac`): [Accelerated Policy Learning with Parallel Differentiable Simulation](http://arxiv.org/abs/2204.07137)
    - **SHA2C** (`algo=sha2c`): Short-Horizon Asymmetric Actor-Critic

### Dynamical models

We have implemented four types of dynamic models for the quadrotor:
- **Full Quadrotor Dynamics** (`dynamics=quad`): Simulates the full dynamics of the quadrotor, including the aerodynamic effect, as described in [Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](http://arxiv.org/abs/2305.02772).
- **(TODO) Simplified Quadrotor Dynamics** (`dynamics=simple`): Simulates the attitude dynamics of the quadrotor, but without considering body rate dynamics, as described in [Learning Quadrotor Control From Visual Features  Using Differentiable Simulation](http://arxiv.org/abs/2410.15979).
- **Discrete Point Mass Dynamics** (`dynamics=pmd`): Simulates the quadrotor as a point mass, ignoring its pose for faster simulation and smoother gradient flow, as described in [Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics](http://arxiv.org/abs/2407.10648).
- **Continuous Point Mass Dynamics** (`dynamics=pmc`): Simulates the quadrotor as a point mass, ignoring its pose, but with continuous time integration.

### Sensors
DiffAero supports two types of exteroceptive sensors:
- **Depth Camera** (`sensor=camera`): Provides depth information about the environment.
- **LiDAR** (`sensor=lidar`): Provides distance measurements to nearby obstacles.

## Installation

### System requirements

- System: Ubuntu.
- Pytorch 2.x.

### Installing the DiffAero

Clone this repo and install the python package:

```bash
git clone https://github.com/zxh0916/diffaero.git
cd diffaero && pip install -e .
```

## Usage

### Basic usage
Under the repo's root directory, run the following command to train a policy (`[a,b,c]` means `a` or `b` or `c`, etc.):

```bash
python script/train.py env=[pc,oa,racing] algo=[apg,apg_sto,shac,sha2c,ppo,world]
```

Note that `env=[pc,oa]` means use `env=pc` or `env=oa`, etc.

Once the training is done, run the following command to test the trained policy:

```bash
python script/test.py env=[pc,oa,racing] checkpoint=/absolute/path/to/checkpoints/directory use_training_cfg=True n_envs=64
```

To list all configuration choices, run:

```bash
python script/train.py -h
```

To enable tab-completion in command line, run:
```bash
eval "$(python script/train.py -sc install=bash)"
```

### Visualization

#### Visualization with taichi GUI

DiffAero supports real-time visualization using [taichi GGUI system](https://docs.taichi-lang.org/docs/ggui). To enable the GUI, set `headless=False` in the training or testing command. Note that the taichi GUI can only be used  with GPU0 (`device=0`) on workstation with multiple GPUs. For example, to visualize the training process of the Position Control task, run:
```bash
python script/train.py env=pc headless=False device=0
```

#### Visualize the depth camera and LiDAR data

To visualize the depth camera and LiDAR data in the Obstacle Avoidance task, set `display_image=True` in the training or testing command. For example, to visualize the depth camera data during testing, run:
```bash
python script/train.py env=oa display_image=True
```

#### Record First-Person View Videos

The Obstacle Avoidance task supports recording first-person view videos from the quadrotor's first-person perspective. To record videos, set `record_video=True` in the testing command:
```bash
python script/train.py env=oa checkpoint=/absolute/path/to/checkpoints/directory use_training_cfg=True n_envs=16 record_video=True
```
The recorded videos will be saved in the `outputs/test/YYYY-MM-DD/HH-MM/video` directory under the repo's root directory.a

### Sweep across multiple configurations

DiffAero supports sweeping across multiple configurations using [hydra](https://hydra.cc). For example, you can specify multiple values to one argument by separating them with commas, and hydra will automatically generate all combinations of the specified values. For example, to sweep across different environments and algorithms, you can run:
```bash
python script/train.py -m env=pc,oa,racing algo=apg,apg_sto,shac,sha2c,ppo,world # generate 3x6=18 combinations, executed sequentially
```

#### Sweep across multiple GPUs in parallel

For workstations with multiple GPUs, you can specify multiple devices by setting `device` to string containing multiple GPU indices and setting `n_jobs` greater than 1 to sweep through configuation combinations in parallel using [hydra-joblib-launcher](https://hydra.cc/docs/plugins/joblib_launcher/) and [joblib](https://joblib.readthedocs.io/en/stable/). For example, to use the first 4 GPUs (GPU0, GPU1, GPU2, GPU3), run:
```bash
# generate 2x2x3=12 combinations, executed in parallel on 4 GPUs, with 3 jobs each
python script/train.py -m env=pc,oa algo=apg_sto,shac algo.l_rollout=16,32,64 n_jobs=4 device="0123" 
```

#### Automatic Hyperparameter Tuning

DiffAero supports automatic hyperparameter tuning using [hydra-optuna-sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/) and [Optuna](https://optuna.org/). To search for the hyperparameter configuration that maximizes the success rate, uncomment the `override hydra/sweeper: optuna_sweep` line in `cfg/config_train.yaml`, specify the hyperparameters to be optimized in the `cfg/hydra/sweeper/optuna_sweep.yaml` file, and run
```python
python script/train.py -m
```
This feature can be combined with multi-device parallel sweep to further speed up the hyperparameter search.

## Deploy

If you want to evaluate and deploy your trained policy in Gazebo or in real world, please refer to this repository (Coming soon).

## TODO-List
- [ ] Add simplified quadrotor dynamics model.
- [ ] Add support to train policies with [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (maybe).
- [ ] Update the LiDAR sensor to be more realistic.

## Citation

If you find DiffAero useful in your research, please consider citing:

```bibtex
@misc{zhang2025diffaero,
      title={DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning}, 
      author={Xinhong Zhang and Runqing Wang and Yunfan Ren and Jian Sun and Hao Fang and Jie Chen and Gang Wang},
      year={2025},
      eprint={2509.10247},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.10247}, 
}
```
