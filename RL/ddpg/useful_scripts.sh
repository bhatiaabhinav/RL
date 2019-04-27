# $GYM_PYTHON -m RL.ddpg.addpg_solver --env=$1 --seed=0 --test_seed=42 --ob_dtype=float32 --nstack=1 --nn_size="[400,300]" --rms_norm_action=True --tau=0.001 --gamma=0.99 --exploration_episodes=10 --use_param_noise=True --exploration_sigma=0.2 --training_episodes=10000 --mb_size=32 --init_scale=3e-3 --lr=1e-3 --a_lr=1e-4 --l2_reg=1e-2 --train_every=2 --exploit_every=4 --logger_level=INFO --use_layer_norm=True --run_no_prefix=ddpg-0 --render=False

$GYM_PYTHON -m RL.ddpg.addpg_solver --env=$1 --seed=0 --test_seed=42 --ob_dtype=float32 --nstack=1 --nn_size="[400,300]" --rms_norm_action=True --tau=0.001 --gamma=0.99 --exploration_episodes=10 --use_param_noise=True --exploration_sigma=0.2 --training_episodes=10000 --mb_size=32 --init_scale=3e-3 --lr=1e-3 --a_lr=1e-4 --l2_reg=1e-2 --train_every=2 --exploit_every=4 --logger_level=INFO --use_layer_norm=True --run_no_prefix=ddpg-0 --render=True --render_graphs=False --test_mode=True --saved_model="C:\\Users\\abhinavb\\workspace\\logs\\HalfCheetah-v2\\ddpg-0_001\\model"