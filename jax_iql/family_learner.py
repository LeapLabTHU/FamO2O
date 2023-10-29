"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax

import family_temperature
import policy
import value_net
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v
from family_actor import family_hierarchical_update as family_update_hierarchical_actor
from family_actor import family_update as family_awr_update_actor
from family_utils import sin_cos_skill_func
from learner import target_update


@partial(jax.jit, static_argnames=(
        'random_hierarchical_coefficients', "family_sin_cos_n", "family_sin_cos_d"))
def _family_update_jit(
        rng: PRNGKey,
        actor: Model, hierarchical_actor: Model, temp: Model,
        critic: Model, value: Model, target_critic: Model,
        batch: Batch,
        discount: float, tau: float, expectile: float, target_entropy: float,
        family_coefficient_range: Sequence[float],
        family_sin_cos_n: float, family_sin_cos_d: int,
        random_hierarchical_coefficients: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, InfoDict]:
    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)

    new_hierarchical_actor, hierarchical_actor_info = family_update_hierarchical_actor(
        key=key, hierarchical_actor=hierarchical_actor, actor=actor, critic=target_critic, temp=temp, batch=batch,
        family_coefficient_range=family_coefficient_range, family_sin_cos_n=family_sin_cos_n,
        family_sin_cos_d=family_sin_cos_d)

    new_temp, alpha_info = family_temperature.update(temp=temp, entropy=hierarchical_actor_info['hierarchical_entropy'],
                                                     target_entropy=target_entropy)

    key, rng = jax.random.split(rng)

    new_actor, actor_info = family_awr_update_actor(
        key=key, actor=actor, hierarchical_actor=hierarchical_actor, critic=target_critic,
        value=new_value, batch=batch, random_hierarchical_coefficients=random_hierarchical_coefficients,
        family_coefficient_range=family_coefficient_range,
        family_sin_cos_n=family_sin_cos_n, family_sin_cos_d=family_sin_cos_d,
    )

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_hierarchical_actor, new_temp, new_actor, new_value, new_critic, new_target_critic, {
        **hierarchical_actor_info,
        **alpha_info,
        **actor_info,
        **critic_info,
        **value_info,
    }


class FamilyLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 # newly add
                 family_sin_cos_n: float = 10000.0,
                 family_sin_cos_d: int = 6,
                 family_coefficient_range: Sequence[float] = (1.0, 5.0),
                 target_entropy: Optional[float] = None,
                 hierarchical_init_temperature: float = 1.0,
                 hierarchical_temp_lr: float = 3e-4,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.family_sin_cos_n = family_sin_cos_n
        self.family_sin_cos_d = family_sin_cos_d
        self.family_coefficient_range = family_coefficient_range

        if target_entropy is None:
            self.target_entropy = -1.0 / 2.0
        else:
            self.target_entropy = target_entropy

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, hierarchical_actor_key, critic_key, value_key, hierarchical_temp_key = jax.random.split(rng, 6)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims=hidden_dims,
                                            action_dim=action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        # TODO: search whether state dependent std or not
        hierarchical_actor_def = policy.NormalTanhPolicy(hidden_dims=hidden_dims,
                                                         action_dim=1,
                                                         log_std_scale=1.0,
                                                         state_dependent_std=False,
                                                         tanh_squash_distribution=True)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        hierarchical_optimiser = optax.adam(learning_rate=actor_lr)

        observations_for_actor = jnp.concatenate([observations,
                                                  jnp.zeros((observations.shape[0], self.family_sin_cos_d))],
                                                 axis=-1)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations_for_actor],
                             tx=optimiser)
        hierarchical_actor = Model.create(hierarchical_actor_def,
                                          inputs=[hierarchical_actor_key, observations],
                                          tx=hierarchical_optimiser)
        hierarchical_temp = Model.create(family_temperature.Temperature(hierarchical_init_temperature),
                                         inputs=[hierarchical_temp_key],
                                         tx=optax.adam(learning_rate=hierarchical_temp_lr))

        critic_def = value_net.DoubleCritic(hidden_dims)
        value_def = value_net.ValueCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng
        self.hierarchical_actor = hierarchical_actor
        self.hierarchical_temp = hierarchical_temp

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        if observations.ndim == 1:
            observations = observations[None]

        rng, hierarchical_coefficients = policy.sample_actions(
            self.rng, self.hierarchical_actor.apply_fn,
            self.hierarchical_actor.params, observations,
            temperature,
            deterministic=True)
        self.rng = rng
        # hierarchical_coefficients = jnp.clip(hierarchical_coefficients, -1, 1)
        hierarchical_coefficients = (hierarchical_coefficients + 1.0) / 2.0 * (
                self.family_coefficient_range[1] - self.family_coefficient_range[0]) + self.family_coefficient_range[0]
        hierarchical_sin_cos_coefficients = sin_cos_skill_func(
            hierarchical_coefficients, n=self.family_sin_cos_n, d=self.family_sin_cos_d)
        hierarchical_sin_cos_coefficients = np.asarray(hierarchical_sin_cos_coefficients)
        observations_for_policy = np.concatenate([observations, hierarchical_sin_cos_coefficients], axis=-1)

        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations_for_policy,
                                             temperature, deterministic=False)
        self.rng = rng

        actions = np.asarray(actions)
        if actions.ndim > 1:
            actions = actions.squeeze()

        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, random_hierarchical_coefficients: bool) -> InfoDict:
        """
        rng: PRNGKey,
        actor: Model, hierarchical_actor: Model, temp: Model,
        critic: Model, value: Model, target_critic: Model,
        batch: Batch,
        discount: float, tau: float, expectile: float, target_entropy: float,
        family_coefficient_range: Sequence[float],
        family_sin_cos_n: float, family_sin_cos_d: int,
        random_hierarchical_coefficients: bool,
        """
        new_rng, new_hierarchical_actor, new_temp, new_actor, new_value, new_critic, new_target_critic, info = \
            _family_update_jit(
                rng=self.rng,
                actor=self.actor, hierarchical_actor=self.hierarchical_actor, temp=self.hierarchical_temp,
                critic=self.critic, value=self.value, target_critic=self.target_critic,
                batch=batch,
                discount=self.discount, tau=self.tau, expectile=self.expectile, target_entropy=self.target_entropy,
                family_coefficient_range=self.family_coefficient_range, family_sin_cos_n=self.family_sin_cos_n,
                family_sin_cos_d=self.family_sin_cos_d,
                random_hierarchical_coefficients=random_hierarchical_coefficients,
            )

        self.rng = new_rng
        self.hierarchical_actor = new_hierarchical_actor
        self.hierarchical_temp = new_temp
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
