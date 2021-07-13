# RL

This repository is very old and not maintained anymore. I now maintain https://github.com/bhatiaabhinav/RL-v2 for Python+Pytorch implementation of RL algorithms, and https://github.com/bhatiaabhinav/RL.jl for Julia+Flux implementation.


Some algorithms for solving RL problems

## Installation

```sh
git clone https://github.com/bhatiaabhinav/RL.git
cd RL
python -m virtualenv env --python=python3
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Environment variables

`OPENAI_LOGDIR` env variable should point to the directory where you want logs to be written to.

### To run an algorithm

Launch the corresponding module from RL/algorithms folder. `--env_id=<gym_env_id>` is a compulsory parameter. 
Specifiable hyperparameters are listed in RL/core/context.py .

Example Scripts:

After activating the virtualenv:

For DQN:
```bash
python -m RL.algorithms.simple_dqn --env_id=BreakoutNoFrameskip-v4 --experiment_name=SimpleDQN --double_dqn=False --dueling_dqn=False --experience_buffer_length=100000 --atari_clip_rewards=False --atari_episode_life=True --learning_rate=1e-4 --convs="[(16,8,4),(32,4,2),(32,3,1)]" --hidden_layers="[256]" --normalize_observations=False --minimum_experience=10000 --target_network_update_every=2000
```

For SAC:
```bash
python -m RL.algorithms.sac --env_id=HalfCheetah-v2 --experiment_name=sac --num_steps_to_run=500000 --normalize_observations=False --alpha=0.2 --actor_learning_rate=0.0003 --learning_rate=0.0003 --target_network_update_tau=0.005 --exploit_every=8 --minimum_experience=10000 --logstd_min=-20 --logstd_max=2 --num_critics=2 --init_scale=None --l2_reg=0 --train_every=1 --experience_buffer_length=1000000 --minibatch_size=100 --hidden_layers=[256,256] --gamma=0.99 --layer_norm=False --clip_gradients=None --record_returns=False --reward_scaling=1 --ignore_done_on_timelimit=True
```
