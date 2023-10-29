from JaxCQL.conservative_sac import ConservativeSAC

config = dict(
    max_traj_length=1000,

    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=(-400, 200),  # negative for offline and positive for online
    num_train_loop_per_epoch=5,
    num_trains_per_train_loop=1000,
    num_expl_steps_per_train_loop=1000,

    bc_epochs=0,
    eval_period=1,
    eval_n_trajs=5,

    replay_buffer_max_size=int(2e6),

    cql=ConservativeSAC.get_default_config(),
)
