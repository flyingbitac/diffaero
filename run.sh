#!/bin/bash

python script/train.py -m algo=shac env=oa algo.l_rollout=16,20,32,48 algo.actor_lr=0.003,0.001,0.0003,0.0001 algo.critic_lr=0.003,0.001,0.0003,0.0001 n_updates=5000