from functools import partial
from typing import Tuple, Sequence

import jax.numpy as jnp
import jax.random

from common import Batch, InfoDict, Model, Params, PRNGKey
from family_utils import sin_cos_skill_func


@partial(jax.jit, static_argnames=("family_sin_cos_n", 'family_sin_cos_d'))
def family_hierarchical_update(
        key: PRNGKey, actor: Model, hierarchical_actor: Model, critic: Model, temp: Model,
        batch: Batch,
        family_coefficient_range: Sequence[float], family_sin_cos_n: float, family_sin_cos_d: int,
) -> Tuple[Model, InfoDict]:
    def hierarchical_actor_loss_fn(hierarchical_actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        hierarchical_dist = hierarchical_actor.apply({"params": hierarchical_actor_params},
                                                     batch.observations,
                                                     training=True,
                                                     rngs={"dropout": key},
                                                     )
        hierarchical_coefficients = hierarchical_dist.sample(seed=key)
        hierarchical_log_probs = hierarchical_dist.log_prob(hierarchical_coefficients)
        hierarchical_coefficients = (hierarchical_coefficients + 1.0) / 2.0 * (
                family_coefficient_range[1] - family_coefficient_range[0]) + \
                                    family_coefficient_range[0]
        hierarchical_sin_cos_coefficients = sin_cos_skill_func(
            hierarchical_coefficients, n=family_sin_cos_n, d=family_sin_cos_d)
        observation_for_policy = jnp.concatenate([batch.observations, hierarchical_sin_cos_coefficients], axis=-1)
        dist = actor.apply({'params': actor.params},
                           observation_for_policy,
                           training=False)
        actions = dist.sample(seed=key)
        q1_hierarchical, q2_hierarchical = critic.apply({'params': critic.params}, batch.observations, actions)
        q_hierarchical = jnp.minimum(q1_hierarchical, q2_hierarchical)

        hierarchical_loss = (hierarchical_log_probs * temp() - q_hierarchical).mean()
        return hierarchical_loss, {
            "hierarchical_loss": hierarchical_loss,
            "q1_hierarchical": q1_hierarchical,
            "q2_hierarchical": q2_hierarchical,
            "q_hierarchical": q_hierarchical,
            "hierarchical_coefficients": hierarchical_coefficients,
            'hierarchical_entropy': -hierarchical_log_probs.mean()
        }

    new_hierarchical_actor, info = hierarchical_actor.apply_gradient(hierarchical_actor_loss_fn)
    return new_hierarchical_actor, info


@partial(jax.jit, static_argnames=(
        'random_hierarchical_coefficients', 'family_sin_cos_n', 'family_sin_cos_d'))
def family_update(
        key: PRNGKey, actor: Model, hierarchical_actor: Model, critic: Model, value: Model,
        batch: Batch, random_hierarchical_coefficients: bool,
        family_coefficient_range: Sequence[float], family_sin_cos_n: float, family_sin_cos_d: int,
) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    if random_hierarchical_coefficients:
        hierarchical_coefficients = jax.random.uniform(key=key, shape=(batch.observations.shape[0], 1),
                                                       minval=family_coefficient_range[0],
                                                       maxval=family_coefficient_range[1])
    else:
        hierarchical_dist = hierarchical_actor.apply({"params": hierarchical_actor.params},
                                                     batch.observations,
                                                     training=False, )

        hierarchical_coefficients = hierarchical_dist.sample(seed=key)
        hierarchical_coefficients = (hierarchical_coefficients + 1.0) / 2.0 * (
                family_coefficient_range[1] - family_coefficient_range[0]) + family_coefficient_range[0]
        hierarchical_coefficients = jax.lax.stop_gradient(hierarchical_coefficients)
    hierarchical_sin_cos_coefficients = sin_cos_skill_func(hierarchical_coefficients, n=family_sin_cos_n,
                                                           d=family_sin_cos_d)
    observation_for_policy = jnp.concatenate([batch.observations, hierarchical_sin_cos_coefficients], axis=-1)

    exp_a = jnp.exp((q - v) * hierarchical_coefficients)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           observation_for_policy,
                           training=True,
                           rngs={'dropout': key})

        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
