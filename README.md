# DiffAero: A GPU-Accelerated Differentiable Simulation Framework for Efficient Quadrotor Policy Learning

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
    - [Recording first-person view videos](#recording-first-person-view-videos)
    - [Sweep across multiple configurations](#sweep-across-multiple-configurations)
      - [Sweep across multiple devices](#sweep-across-multiple-devices)
  - [Deploy](#deploy)

## Introduction

DiffAero is a GPU-accelerated differentiable simulator for quadrotor dynamics and sensor simulation. DiffAero enables flexible and efficient single- and multi-agent flight policy learning by supporting different types of self-defined dynamical models and flight tasks, and rendering sensory data with ray casting methods, all of which implemented in [Pytorch](https://www.pytorch.org) so that they run efficiently on modern GPUs.

DiffAero utilizes a modular design where different components (e.g., sensors, flight tasks, dynamics, network architecture, and learning algorithms) are decoupled from each other and can be configured independently. As a result, users can combine different components almost arbitrarily to initiate a custom-configured training process with minimal effort.

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
- **Obstacle Avoidance** (`env=oa`): The goal is to must navigate to and hover on target positions while avoiding collision with environmental obstacles and other quadrotors, given exteroceptive informations:
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

We have implemented three types of dynamic models for the quadrotor:
- **Full Quadrotor Dynamics** (`dynamics=quad`): Simulates the full dynamics of the quadrotor, including the aerodynamic effects and the motor dynamics(TODO), as described in [Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](http://arxiv.org/abs/2305.02772).
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
cd /path/to/your/workspace
git clone https://github.com/zxh0916/diffaero.git
cd diffaero && pip install -e .
```

## Usage

### Basic usage
Under the repo's root directory, run the following command to train a policy (`[a,b,c]` means `a` or `b` or `c`, etc.):

```bash
python script/train.py env=[pc,oa,racing] algo=[apg,apg_sto,shac,sha2c,ppo,world] n_envs=4096 algo.l_rollout=32 headless=True
```

Note that `env=[pc,oa]` means use `env=pc` or `env=oa`, etc.

Once the training is done, run the following command to test and visualize the trained policy:

```bash
python script/test.py env=[pc,oa,racing] checkpoint=/absolute/path/to/checkpoints/directory use_training_cfg=True n_envs=64 headless=False
```

If you want to see all configuration choices, run:

```bash
python script/train.py -h
```

To enable tab-completion in cli, run:
```bash
eval "$(python script/train.py -sc install=bash)"
```

### Recording first-person view videos

The Obstacle Avoidance task supports recording first-person view videos from the quadrotor's perspective. To record videos, set `headless=True` and `record_video=True` in the training or testing script. The recorded videos will be saved in the `outputs` directory under the repo's root directory.
```bash
python script/train.py env=oa checkpoint=/absolute/path/to/checkpoints/directory use_training_cfg=True n_envs=16 headless=True record_video=True
```

### Sweep across multiple configurations

DiffAero supports sweeping across multiple configurations using [hydra](https://hydra.cc) and [joblib](https://joblib.readthedocs.io/en/stable/). For example, you can specify multiple values to one argument by separating them with commas, and hydra will automatically generate all combinations of the specified values. For example, to sweep across different environments and algorithms, you can run:
```bash
python script/train.py -m env=pc,oa,racing algo=apg,apg_sto,shac,sha2c,ppo,world # generate 3x6=18 combinations, executed sequentially
```

#### Sweep across multiple devices

For workstations with multiple GPUs, you can specify multiple devices and `n_jobs` greater than 1 to sweep through configuation combinations in parallel. For example, to use the first 4 GPUs (GPU0, GPU1, GPU2, GPU3), you can run:
```bash
# generate 2x2x3=12 combinations, executed in parallel on 4 GPUs, with 3 jobs each
python script/train.py -m env=pc,oa algo=apg_sto,shac algo.l_rollout=16,32,64 n_jobs=4 device="0123" 
```

## Deploy

If you want to evaluate and deploy your trained policy in Gazebo or in real world, please refer to [this repository](https://github.com/zxh0916/diffaero-deploy).