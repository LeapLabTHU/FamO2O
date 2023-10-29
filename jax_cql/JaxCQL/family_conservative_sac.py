from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from cql_finetune.family_utils import sin_cos_skill_func
from .conservative_sac import ConservativeSAC
from .jax_utils import (
    next_rng, value_and_multi_grad, mse_loss, JaxRNG, wrap_function_with_rng,
    collect_jax_metrics
)
from .model import Scalar, update_target_network


class FamilyConservativeSAC(ConservativeSAC):
    def __init__(self, config, policy, qf, hierarchical_policy):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        policy_params = self.policy.init(
            next_rng(self.policy.rng_keys()),
            jnp.zeros((10, self.observation_dim + self.config.sin_cos_d)),
        )
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        qf1_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        self._train_states['qf1'] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        self._train_states['qf2'] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})

        model_keys = ['policy', 'qf1', 'qf2']

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states['log_alpha'] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha')

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states['log_alpha_prime'] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha_prime')

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

        self.hierarchical_policy = hierarchical_policy
        hierarchical_policy_params = self.hierarchical_policy.init(
            next_rng(self.hierarchical_policy.rng_keys()),
            jnp.zeros((10, self.observation_dim))
        )

        self._train_states['hierarchical_policy'] = TrainState.create(
            params=hierarchical_policy_params,
            tx=optimizer_class(self.config.hierarchical_policy_lr),
            apply_fn=None
        )
        self._model_keys = tuple(list(self.model_keys) + ['hierarchical_policy', ])

    def train(self, batch, use_cql, random_hierarchical, bc=False):
        self._total_steps += 1
        # print(f'{random_hierarchical=}')
        self._train_states, self._target_qf_params, metrics = self._train_step(
            train_states=self._train_states,
            target_qf_params=self._target_qf_params,
            rng=next_rng(),
            batch=batch,
            use_cql=use_cql,
            random_hierarchical=random_hierarchical,
            bc=bc
        )
        return metrics

    @partial(jax.jit, static_argnames=('self', 'use_cql', 'random_hierarchical', 'bc'))
    def _train_step(self, train_states, target_qf_params, rng, batch, use_cql, random_hierarchical, bc=False):
        rng_generator = JaxRNG(rng)

        def loss_fn(train_params):
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            loss_collection = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_hierarchical_policy(rng, *args, **kwargs):
                return self.hierarchical_policy.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.hierarchical_policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_random_hierarchical_policy(rng, *args, **kwargs):
                coefficient_lower_bound, coefficient_upper_bound = self.hierarchical_policy.coefficient_range
                hierarchical_coefficient = jax.random.uniform(
                    key=rng,
                    shape=(kwargs['observations'].shape[0], 1),
                    minval=coefficient_lower_bound,
                    maxval=coefficient_upper_bound
                )
                return hierarchical_coefficient, None

            @wrap_function_with_rng(rng_generator())
            def forward_qf(rng, *args, **kwargs):
                return self.qf.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.qf.rng_keys())
                )

            hierarchical_coefficient, _ = forward_hierarchical_policy(
                train_params['hierarchical_policy'], observations=observations)
            hierarchical_sin_cos_coefficient = sin_cos_skill_func(
                hierarchical_coefficient,
                n=self.config.sin_cos_n,
                d=self.config.sin_cos_d,
            )

            if not random_hierarchical:
                observations_for_policy = jnp.concatenate([observations, hierarchical_sin_cos_coefficient], axis=-1)
                new_actions, log_pi = forward_policy(train_params['policy'], observations_for_policy)
            else:
                random_hierarchical_coefficient, _ = forward_random_hierarchical_policy(
                    train_params['hierarchical_policy'], observations=observations
                )
                random_hierarchical_sin_cos_coefficient = sin_cos_skill_func(
                    random_hierarchical_coefficient,
                    n=self.config.sin_cos_n,
                    d=self.config.sin_cos_d,
                )
                observations_for_policy = jnp.concatenate(
                    [observations, random_hierarchical_sin_cos_coefficient], axis=-1)
                new_actions, log_pi = forward_policy(
                    train_params['policy'],
                    observations_for_policy)

                not_random_observations_for_policy = jnp.concatenate([observations, hierarchical_sin_cos_coefficient],
                                                                     axis=-1)
                not_random_new_actions, not_random_log_pi = forward_policy(
                    train_params['policy'],
                    not_random_observations_for_policy)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (
                        log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            if bc:
                log_probs = forward_policy(train_params['policy'], observations_for_policy, actions,
                                           method=self.policy.log_prob)
                policy_loss = (alpha * log_pi - log_probs).mean()
                hierarchical_loss = 0.0
            else:
                q_new_actions = jnp.minimum(
                    forward_qf(train_params['qf1'], observations, new_actions),
                    forward_qf(train_params['qf2'], observations, new_actions),
                )
                if not random_hierarchical:
                    policy_loss = (alpha * log_pi - hierarchical_coefficient.squeeze() * q_new_actions).mean()
                    hierarchical_loss = -q_new_actions.mean()
                else:
                    policy_loss = (alpha * log_pi - random_hierarchical_coefficient.squeeze() * q_new_actions).mean()
                    not_random_q_new_actions = jnp.minimum(
                        forward_qf(train_params['qf1'], observations, not_random_new_actions),
                        forward_qf(train_params['qf2'], observations, not_random_new_actions),
                    )
                    hierarchical_loss = -not_random_q_new_actions.mean()

            loss_collection['policy'] = policy_loss
            loss_collection['hierarchical_policy'] = hierarchical_loss

            """ Q function loss """
            q1_pred = forward_qf(train_params['qf1'], observations, actions)
            q2_pred = forward_qf(train_params['qf2'], observations, actions)

            if self.config.cql_max_target_backup:
                next_observations_for_policy = jnp.concatenate([next_observations, hierarchical_sin_cos_coefficient],
                                                               axis=-1)
                new_next_actions, next_log_pi = forward_policy(
                    train_params['policy'], next_observations_for_policy, repeat=self.config.cql_n_actions
                )
                target_q_values = jnp.minimum(
                    forward_qf(target_qf_params['qf1'], next_observations, new_next_actions),
                    forward_qf(target_qf_params['qf2'], next_observations, new_next_actions),
                )
                max_target_indices = jnp.expand_dims(jnp.argmax(target_q_values, axis=-1), axis=-1)
                target_q_values = jnp.take_along_axis(target_q_values, max_target_indices, axis=-1).squeeze(-1)
                next_log_pi = jnp.take_along_axis(next_log_pi, max_target_indices, axis=-1).squeeze(-1)
            else:
                next_observations_for_policy = jnp.concatenate([next_observations, hierarchical_sin_cos_coefficient],
                                                               axis=-1)
                new_next_actions, next_log_pi = forward_policy(
                    train_params['policy'], next_observations_for_policy
                )
                target_q_values = jnp.minimum(
                    forward_qf(target_qf_params['qf1'], next_observations, new_next_actions),
                    forward_qf(target_qf_params['qf2'], next_observations, new_next_actions),
                )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(
                rewards + (1. - dones) * self.config.discount * target_q_values
            )
            qf1_loss = mse_loss(q1_pred, td_target)
            qf2_loss = mse_loss(q2_pred, td_target)

            ### CQL
            if self.config.use_cql and use_cql:
                batch_size = actions.shape[0]
                cql_random_actions = jax.random.uniform(
                    rng_generator(), shape=(batch_size, self.config.cql_n_actions, self.action_dim),
                    minval=-1.0, maxval=1.0
                )

                cql_current_actions, cql_current_log_pis = forward_policy(
                    train_params['policy'], observations_for_policy, repeat=self.config.cql_n_actions,
                )
                cql_next_actions, cql_next_log_pis = forward_policy(
                    train_params['policy'], next_observations_for_policy, repeat=self.config.cql_n_actions,
                )

                cql_q1_rand = forward_qf(train_params['qf1'], observations, cql_random_actions)
                cql_q2_rand = forward_qf(train_params['qf2'], observations, cql_random_actions)
                cql_q1_current_actions = forward_qf(train_params['qf1'], observations, cql_current_actions)
                cql_q2_current_actions = forward_qf(train_params['qf2'], observations, cql_current_actions)
                cql_q1_next_actions = forward_qf(train_params['qf1'], observations, cql_next_actions)
                cql_q2_next_actions = forward_qf(train_params['qf2'], observations, cql_next_actions)

                cql_cat_q1 = jnp.concatenate(
                    [cql_q1_rand, jnp.expand_dims(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], axis=1
                )
                cql_cat_q2 = jnp.concatenate(
                    [cql_q2_rand, jnp.expand_dims(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], axis=1
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5 ** self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [cql_q1_rand - random_density,
                         cql_q1_next_actions - cql_next_log_pis,
                         cql_q1_current_actions - cql_current_log_pis],
                        axis=1
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [cql_q2_rand - random_density,
                         cql_q2_next_actions - cql_next_log_pis,
                         cql_q2_current_actions - cql_current_log_pis],
                        axis=1
                    )

                cql_qf1_ood = (
                        jax.scipy.special.logsumexp(cql_cat_q1 / self.config.cql_temp, axis=1)
                        * self.config.cql_temp
                )
                cql_qf2_ood = (
                        jax.scipy.special.logsumexp(cql_cat_q2 / self.config.cql_temp, axis=1)
                        * self.config.cql_temp
                )

                """Subtract the log likelihood of data"""
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(self.log_alpha_prime.apply(train_params['log_alpha_prime'])),
                        a_min=0.0, a_max=1000000.0
                    )
                    cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (
                            cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (
                            cql_qf2_diff - self.config.cql_target_action_gap)

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection['log_alpha_prime'] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_loss + cql_min_qf1_loss
                qf2_loss = qf2_loss + cql_min_qf2_loss

            loss_collection['qf1'] = qf1_loss
            loss_collection['qf2'] = qf2_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }
        new_target_qf_params = {}
        new_target_qf_params['qf1'] = update_target_network(
            new_train_states['qf1'].params, target_qf_params['qf1'],
            self.config.soft_target_update_rate
        )
        new_target_qf_params['qf2'] = update_target_network(
            new_train_states['qf2'].params, target_qf_params['qf2'],
            self.config.soft_target_update_rate
        )

        metrics = collect_jax_metrics(
            aux_values,
            ['log_pi', 'policy_loss', 'qf1_loss', 'qf2_loss', 'alpha_loss',
             'alpha', 'q1_pred', 'q2_pred', 'target_q_values', "hierarchical_loss"]
        )

        if self.config.use_cql:
            metrics.update(collect_jax_metrics(
                aux_values,
                ['cql_std_q1', 'cql_std_q2', 'cql_q1_rand', 'cql_q2_rand'
                                                            'cql_qf1_diff', 'cql_qf2_diff', 'cql_min_qf1_loss',
                 'cql_min_qf2_loss', 'cql_q1_current_actions', 'cql_q2_current_actions'
                                                               'cql_q1_next_actions', 'cql_q2_next_actions',
                 'alpha_prime',
                 'alpha_prime_loss'],
                'cql'
            ))

        return new_train_states, new_target_qf_params, metrics
