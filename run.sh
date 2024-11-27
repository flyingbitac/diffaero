#!/bin/bash
python script/train.py -m algo=apg,apg_sto,shac,ppo env=pc,oa n_updates=10 n_envs=64