import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import evaluate
from family_learner import FamilyLearner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')

# newly add
flags.DEFINE_integer('last_eval_interval', 5000, 'Eval interval for the last few steps.')
flags.DEFINE_integer('last_eval_start_steps', 35000, 'Number of last few steps')

flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', int(1e6),
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/family_antmaze_finetune_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

RETURN_MAX: float = None
RETURN_MIN: float = None


def normalize(dataset):
    global RETURN_MAX, RETURN_MIN
    if RETURN_MAX is not None and RETURN_MIN is not None:
        return
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)
    RETURN_MAX, RETURN_MIN = compute_returns(trajs[-1]), compute_returns(trajs[0])


def make_env_and_dataset(
        env_name: str,
        seed: int,
) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        pass 
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    kwargs = dict(FLAGS.config)
    del kwargs['config_name']
    family_agent = FamilyLearner(FLAGS.seed,
                                 env.observation_space.sample()[np.newaxis],
                                 env.action_space.sample()[np.newaxis],
                                 max_steps=FLAGS.max_steps,
                                 **kwargs)

    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i >= 1:
            action = family_agent.sample_actions(observation, )
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        if 'antmaze' in FLAGS.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)
        elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
              or 'hopper' in FLAGS.env_name):
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards / (RETURN_MAX - RETURN_MIN) * 1000.0,
                          masks=batch.masks,
                          next_observations=batch.next_observations)
        update_info = family_agent.update(batch, random_hierarchical_coefficients=(i < 1))

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i, max_bins=500)
            summary_writer.flush()

        if (i >= FLAGS.max_steps - FLAGS.last_eval_start_steps) or (-FLAGS.last_eval_start_steps <= i < 1):
            eval_interval = FLAGS.last_eval_interval
        else:
            eval_interval = FLAGS.eval_interval
        if i == 1 - FLAGS.num_pretraining_steps or i % eval_interval == 0:
            eval_stats = evaluate(family_agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
