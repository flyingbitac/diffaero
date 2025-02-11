# QuadDif: A Fully Pytorch-based Differentiable Quadrotor Simulator for Efficient Policy Learning

- [QuadDif: A Fully Pytorch-based Differentiable Quadrotor Simulator for Efficient Policy Learning](#quaddif-a-fully-pytorch-based-differentiable-quadrotor-simulator-for-efficient-policy-learning)
  - [Introduction](#introduction)
    - [Environments](#environments)
    - [Learning Algorithms](#learning-algorithms)
    - [Dynamic Models](#dynamic-models)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Install the QuadDif](#install-the-quaddif)
  - [Usage](#usage)
  - [Deploy](#deploy)

## Introduction

QuadDif is a fully Pytorch-based differentiable quadrotor simulator that utilizes the parallel computing power of GPUs for efficient policy learning. It supports multiple types of dynamic models, all of which are customizable and implemented in [Pytorch](https://www.pytorch.org) so that they run efficiently on modern GPUs and are fully differentiable.

QuadDif utilizes a modular design where different components (e.g., environment, dynamics, network architecture, and learning algorithms) are decoupled from each other and can be configured independently. As a result, users can combine different components almost arbitrarily to initiate a custom-configured training process with minimal effort.

### Environments

QuadDif now supports two flight tasks: 
- **Position Control**: The goal is to move the quadrotor to a target position and hover stably.
- **Obstacle Avoidance**: The goal is to navigate the quadrotor to a target position while avoiding obstacles along the way, given external perceptional informations:
  - Relative positions of obstacles w.r.t. the quadrotor, or
  - Image from the depth camera attached to the quadrotor, or
  - Ray distance from the LiDAR attached to the quadrotor.

### Learning Algorithms

We have implemented several learning algorithms, including RL algorithms and algorithms that exploit the differentiability of the simulator:

- **Reinforcement Learning algorithms**:
    - **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
    - **Dreamer v3**: [Mastering Diverse Domains through World Models](http://arxiv.org/abs/2301.04104)

- **Differentiable algorithms**:
    - **BPTT**(deterministic policy and stochastic policy)
    - **SHAC**: [Accelerated Policy Learning with Parallel Differentiable Simulation](http://arxiv.org/abs/2204.07137)

### Dynamic Models

We have implemented three types of dynamic models for the quadrotor:
- **Full Quadrotor Dynamics**: Simulates the full dynamics of the quadrotor, including the aerodynamic effects and the motor dynamics(TODO), as described in [Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](http://arxiv.org/abs/2305.02772).
- **Point Mass Dynamics**: Simulates the quadrotor as a point mass, ignoring its pose for faster simulation and smoother gradient flow, as described in [Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics](http://arxiv.org/abs/2407.10648).
- **Simplified Quadrotor Dynamics**: Simplified Quadrotor dynamic model with reduced dimension of state space and complexity, as described in [Learning quadrotor control from visual features using differentiable simulation](http://arxiv.org/abs/2410.15979). (TODO)

## Installation

### Requirements

- System: Ubuntu.
- Pytorch 2.x.

### Install the QuadDif

First, clone the QuadDif under your workspace:

```bash
cd /path/to/your/workspace
git clone https://github.com/zxh0916/quaddif.git
```

Then, install the dependencies:

```bash
cd quaddif
# install pytorch3d from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# install other requirements
pip install -r requirements.txt
```

## Usage

Under the repo's root directory, run:

```bash
python script/train.py env=[pc,oa] algo=[apg,apg_sto,shac,ppo] n_envs=4096 l_rollout=32 headless=True
```

to train the model. Once the training is done, run:

```bash
python script/test.py env=[pc,oa] checkpoint=/absolute/path/to/checkpoints/directory n_envs=64 headless=False
```

to test the model and create a GUI viewer to visualize the environment and the learned policy.

## Deploy

If you want to test your trained policy in a SITL simulation experiment, please refer to [this tutorial](deploy/README.md).