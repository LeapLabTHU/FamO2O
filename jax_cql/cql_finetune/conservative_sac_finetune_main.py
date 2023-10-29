import hashlib
import importlib
from argparse import ArgumentParser

import gym
import d4rl
import numpy as np
from ml_collections import ConfigDict

from JaxCQL.conservative_sac import ConservativeSAC
from JaxCQL.jax_utils import batch_to_jax
from JaxCQL.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from JaxCQL.replay_buffer import get_d4rl_dataset, ReplayBuffer
from JaxCQL.sampler import TrajSampler, StepSampler
from JaxCQL.utils import (
    Timer, set_random_seed, prefix_metrics
)
from cql_finetune.utils import MyWandBLogger
from viskit.logging import logger, setup_logger


def encode(name: str, length=64):
    m = hashlib.md5()
    m.update(str.encode(name))
    return m.hexdigest()[:length]


def get_config():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-name', type=str, required=True, help='The name of config files, e.g., base')
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--wandb-offline', action='store_true')
    parser.add_argument('--wandb-project', type=str)
    parser.add_argument('--wandb-entity', type=str)
    parser.add_argument('--wandb-output-dir', type=str, default='./experiment_output')
    args = parser.parse_args()

    config_module = importlib.import_module(
        f'cql_finetune.finetune_configs.{args.config_name}')
    config = config_module.config
    config.update(vars(args))
    config = ConfigDict(config, convert_dict=True)

    exp_encoding = encode(f'CQL_finetune_{config.env}_{config.config_name}')
    exp_name = f'seed_{config.seed}_{config.env}_{exp_encoding}'
    group_name = f"{config.env}_{exp_encoding}"

    config.logging = MyWandBLogger.get_default_config(updates=dict(
        online=not config.wandb_offline,
        exp_name=exp_name,
        group_name=group_name,
        project=config.wandb_project,
        entity=config.wandb_entity,
        output_dir=config.wandb_output_dir,
    ))

    return config


def main(config):
    wandb_logger = MyWandBLogger(config=config.logging, variant=config)
    setup_logger(
        variant=config.to_dict(),
        exp_id=wandb_logger.experiment_id,
        seed=config.seed,
        base_log_dir=config.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(config.seed)

    eval_sampler = TrajSampler(gym.make(config.env).unwrapped, config.max_traj_length)
    expl_sampler = StepSampler(gym.make(config.env).unwrapped, config.max_traj_length)

    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * config.reward_scale + config.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -config.clip_action, config.clip_action)
    # convert dataset to replay_buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_max_size, data=dataset,
                                 reward_scale=config.reward_scale, reward_bias=config.reward_bias)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, config.policy_arch, config.orthogonal_init,
        config.policy_log_std_multiplier, config.policy_log_std_offset
    )
    qf = FullyConnectedQFunction(observation_dim, action_dim, config.qf_arch, config.orthogonal_init)

    if config.cql.target_entropy >= 0.0:
        config.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(config.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}
    for epoch_num_idx, epoch_num in enumerate(config.n_epochs):
        is_offline = epoch_num < 0
        range_args = (epoch_num, 0, 1) if is_offline else (0, epoch_num, 1)

        offline_switch_to_online = epoch_num_idx > 0 and not is_offline and config.n_epochs[epoch_num_idx - 1] < 0
        if config.clear_replay_buffer_online and offline_switch_to_online:
            replay_buffer.clear()
        if config.sac_online and offline_switch_to_online:
            use_cql = False
        else:
            use_cql = True
        for epoch in range(*range_args):
            metrics = {'epoch': epoch}

            with Timer() as train_timer:
                for train_loop_idx in range(config.num_train_loop_per_epoch):
                    if not is_offline:
                        expl_samples = expl_sampler.sample(
                            sampler_policy.update_params(sac.train_params['policy']),
                            n_steps=config.num_expl_steps_per_train_loop,
                            deterministic=False
                        )
                        replay_buffer.add_batch(expl_samples)

                    for train_idx in range(config.num_trains_per_train_loop):
                        batch = batch_to_jax(replay_buffer.sample(config.batch_size))
                        metrics.update(
                            prefix_metrics(
                                sac.train(batch, use_cql=use_cql, bc=(is_offline and abs(epoch) < config.bc_epochs)),
                                'sac'))

            with Timer() as eval_timer:
                if epoch == 0 or (epoch + 1) % config.eval_period == 0:
                    trajs = eval_sampler.sample(
                        sampler_policy.update_params(sac.train_params['policy']),
                        config.eval_n_trajs, deterministic=True
                    )

                    metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                    metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                    metrics['average_normalizd_return'] = np.mean(
                        [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                    )
                    if config.save_model:
                        save_data = {'sac': sac, 'variant': config, 'epoch': epoch}
                        wandb_logger.save_pickle(save_data, 'model.pkl')

            metrics['train_time'] = train_timer()
            metrics['eval_time'] = eval_timer()
            metrics['epoch_time'] = train_timer() + eval_timer()
            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if config.save_model:
        save_data = {'sac': sac, 'variant': config, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    main(config=get_config())
