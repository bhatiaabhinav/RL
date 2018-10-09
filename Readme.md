# RL

Some algorithms for solving RL problems

## Installation

```sh
pip install git+https://github.com/bhatiaabhinav/gymGame.git
pip install git+https://github.com/bhatiaabhinav/gym-ERSLE.git
pip install git+https://github.com/bhatiaabhinav/gym-BSS.git

git clone https://github.com/bhatiaabhinav/RL.git
cd RL
pip install -e .
```

### Environment variables
Configure `GYM_PYTHON` environment variable to point to the python binary of the installation/virtualenv where all the dependencies are installed.

`OPENAI_LOGDIR` env variable should point to the directory where you want logs to be written to.

## DDPG

- Standard DDPG (without CNNs) (Lillicrap et al) + some enhancements like adaptive parameter noise, obs and action normalization etc.
- It can handle linear constraints in action space.


To run standard ddpg,
```bash
sh scripts/standard_ddpg.sh <env_name>
```

## DQN

- Standard DQN (without CNNs) (Mnih et al) + some enhancements like double DQN, dueling DQN etc.

To run,
```bash
$GYM_PYTHON -m RL.dqn.dqn <env_name>
```
