#!/bin/bash

# test different networks
python script/train.py -m algo=apg,apg_sto,shac,shac_q,ppo env=oa n_updates=10 n_envs=64 network=mlp,cnn,rnn,rcnn

# test different environments
python script/train.py -m algo=apg,apg_sto,shac,shac_q,ppo env=pc,oa n_updates=10 n_envs=64 network=mlp,rnn