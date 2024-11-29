# QuadDif: A Fully Pytorch-based Differentiable Quadrotor Simulator for Efficient Policy Learning

## Introduction

QuadDif is a fully Pytorch-based differentiable quadrotor simulator that leverage the massive parall
for efficient control policy learning. It now support full quadrotor dynamics and simplified point-mass dynamics, both configurable and implemented in [Pytorch](www.pytorch.org), thus can run efficiently on modern GPUs and fully differentiable.

QuadDif is also designed modularly, with different components (e.g. environments, dynamics, network architectures, and learning algorithms) decoupled with each other and configurable independently. Consequently, one can nearly arbitrarily combine different components to start the training process with specific configuration with minimal effort.

### Environments

QuadDif now has two flight tasks: 
- **Position Control**: The goal is to move the quadrotor to a target position and hover stably.
- **Obstacle Avoidance**: The goal is to navigate the quadrotor to a target position while avoiding obstacles along the way, given external perceptional informations:
  - Relative positions of obstacles w.r.t. the quadrotor, or
  - Image from the depth camera attached to the quadrotor, or
  - Ray distance from the LiDAR attached to the quadrotor.

Since the development of a 3D GUI viewer needs a large amont of work, QuadDif is built upon [Nvidia Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) for now, not for physical simulation, but for visualize the environment and the control policy.

### Learning Algorithms

We also implemented several learning algorithms, including RL algorithms and those leverage the differentiability of the simulator:

- Reinforcement Learning algorithms:
    - **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
    - **Dreamer v3**: [Mastering Diverse Domains through World Models](http://arxiv.org/abs/2301.04104)

- Differentiable algorithms:
    - **BPTT**(deterministic policy and stochastic policy)
    - **SHAC**: [Accelerated Policy Learning with Parallel Differentiable Simulation](http://arxiv.org/abs/2204.07137)

### Dynamic Models

We implemented three types of dynamic models for the quadrotor:
- **Full Quadrotor Dynamics**: Simulates the full dynamics of the quadrotor, including the aerodynamic effects and the motor dynamics(TODO), as described in [Efficient and Robust Time-Optimal Trajectory Planning and Control for Agile Quadrotor Flight](http://arxiv.org/abs/2305.02772).
- **Point Mass Dynamics**: Simulates the quadrotor as a point mass, ignoring its attitute, for better performance and gradient flow, as described in [Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics](http://arxiv.org/abs/2407.10648).
- **Simplified Quadrotor Dynamics**: Simplified Quadrotor dynamic model with reduced dimension of state space and complexity, as described in [Learning quadrotor control from visual features using differentiable simulation](http://arxiv.org/abs/2410.15979). (TODO)

## Installation

### Requirements
- Ubuntu 18.04 or 20.04.
- Python 3.6, 3.7 or 3.8.
- Minimum NVIDIA driver version: 470.xx

First, download the Nvidia Isaac Gym Preview 4 from [their website](https://developer.nvidia.com/isaac-gym), and then unpack it to some where.

```bash
tar -xf ./IsaacGym_Preview_4_Package.tar.gz
```

Then, install the Isaac Gym.

```bash
cd isaacgym/python
pip install -e .
```

To verify the details of the installed package, run:

```bash
pip show isaacgym
```

Then, clone the repository under your workspace.

```bash
cd /path/to/your/workspace
git clone https://github.com/zxh0916/quaddif.git
```

Finally, install the dependencies.

```bash
cd quaddif
pip install -r requirements.txt
```

## Usage

Under the repo's root directory, run

```bash
python script/train.py env=[pc,oa] algo=[apg,apg_sto,shac,ppo] n_envs=4096 l_rollout=32 headless=True
```

to train the model. Once the training is done, run

```bash
python script/test.py env=[pc,oa] checkpoint=/absolute/path/to/checkpoints/directory n_envs=64 headless=False
```

to test the model and create a GUI viewer to visualize the environment and the learned policy.