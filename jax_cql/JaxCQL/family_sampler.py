import jax.numpy as jnp
import numpy as np

from cql_finetune.family_utils import sin_cos_skill_func


class FamilyStepSampler(object):

    def __init__(
            self,
            env,
            sin_cos_n,
            sin_cos_d,
            max_traj_length=1000
    ):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

        self.sin_cos_n = sin_cos_n
        self.sin_cos_d = sin_cos_d

    def sample(self, policy, hierarchical_policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            hierarchical_coefficient = hierarchical_policy(observation.reshape(1, -1), deterministic=True)
            hierarchical_sin_cos_coefficient = sin_cos_skill_func(
                hierarchical_coefficient,
                n=self.sin_cos_n,
                d=self.sin_cos_d,
            )
            observations_for_policy = jnp.concatenate([observation.reshape(1, -1), hierarchical_sin_cos_coefficient],
                                                      axis=-1)
            action = policy(observations_for_policy.reshape(1, -1), deterministic=deterministic).reshape(-1)
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class FamilyTrajSampler(object):

    def __init__(
            self,
            env,
            sin_cos_n,
            sin_cos_d,
            max_traj_length=1000
    ):
        self.max_traj_length = max_traj_length
        self._env = env
        self.sin_cos_n = sin_cos_n
        self.sin_cos_d = sin_cos_d

    def sample(self, policy, hierarchical_policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            observation = self.env.reset()

            for _ in range(self.max_traj_length):
                hierarchical_coefficient = hierarchical_policy(observation.reshape(1, -1), deterministic=True)
                hierarchical_sin_cos_coefficient = sin_cos_skill_func(
                    hierarchical_coefficient,
                    n=self.sin_cos_n,
                    d=self.sin_cos_d,
                )
                observations_for_policy = jnp.concatenate(
                    [observation.reshape(1, -1), hierarchical_sin_cos_coefficient], axis=-1)

                action = policy(observations_for_policy.reshape(1, -1), deterministic=deterministic).reshape(-1)
                next_observation, reward, done, _ = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))

        return trajs

    @property
    def env(self):
        return self._env
