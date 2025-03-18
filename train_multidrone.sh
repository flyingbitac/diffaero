algo="mashac"
env="mapc"
n_agents=3
n_envs=4096
l_rollout=32
headless=True
collide_loss_coef=16.0
actor_lr=0.001
critic_lr=0.001
n_updates=1200
use_old_obs_proc=1
target_update_rate=0.5

# seed_list=(1 2 3 4 5)
seed_list=(1)

for seed in "${seed_list[@]}";
    do  {
        current_time=$(date +"%H-%M-%S")
        seed=$((seed))
        echo "seed is ${seed}:"
        CUDA_DEVICE=$((seed % 4))
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python script/train.py env=${env} algo=${algo} n_envs=${n_envs} l_rollout=${l_rollout} headless=${headless} n_agents=${n_agents} \
        n_updates=${n_updates} seed=${seed} algo.actor_lr=${actor_lr} algo.critic_lr=${critic_lr} +current_time=${current_time} +env.collide_loss_coef=${collide_loss_coef} \
        +env.use_old_obs_proc=${use_old_obs_proc} algo.target_update_rate=${target_update_rate}
        sleep 3
    } &
sleep 3
done

sleep 6
wait