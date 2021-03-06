$GYM_PYTHON -m RL.ddpg.addpg_solver \
	--env=$1 \
	--seed=0 \
	--test_seed=42 \
	--ob_dtype=float32 \
	--nstack=3 \
	--nn_size="[128,96]" \
	--soft_constraints=True \
	--soft_constraints_lambda=1000 \
	--softmax_actor=False \
	--wolpertinger_critic_train=True \
	--log_norm_obs_alloc=False \
	--log_norm_action=False \
	--rms_norm_action=True \
	--tau=0.001 \
	--gamma=1 \
	--exploration_episodes=10 \
	--use_param_noise=True \
	--use_safe_noise=False \
	--exploration_theta=1 \
	--training_episodes=20000 \
	--mb_size=128 \
	--init_scale=3e-3 \
	--lr=1e-3 \
	--a_lr=1e-4 \
	--l2_reg=1e-2 \
	--train_every=2 \
	--exploit_every=4 \
	--logger_level=INFO \
	--use_batch_norm=False \
	--use_layer_norm=True \
	--run_no_prefix=$2_test \
	--test_mode=True \
	--saved_model=$OPENAI_LOGDIR/$1/$2/model \
	--monitor=False \

# this file was autogenerated by running create_test.sh variants/sddpg_rmsac_wolpert.sh
