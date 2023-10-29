from typing import Tuple, List, Union

import distrax
import jax.numpy as jnp

from JaxCQL.model import TanhGaussianPolicy, FullyConnectedNetwork, Scalar
from .jax_utils import extend_and_repeat, next_rng, JaxRNG


class HierarchicalPolicy(TanhGaussianPolicy):
    coefficient_range: Union[Tuple, List] = (0.5, 1.5)

    observation_dim: int
    action_dim: int = 1
    arch: str = '256-256'
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0

    def setup(self):
        if len(self.coefficient_range) != 2:
            raise ValueError(f'Invalid coefficient_range: {len(self.coefficient_range)}')
        super(HierarchicalPolicy, self).setup()

    def log_prob(self, observations, actions):
        # if actions.ndim != 1:
        #     raise ValueError(f'Invalid action dim: {actions.ndim}')
        transformed_actions = (actions - self.coefficient_range[0]) / (
                self.coefficient_range[1] - self.coefficient_range[0])
        return super(HierarchicalPolicy, self).log_prob(
            observations=observations, actions=transformed_actions)

    def __call__(self, observations, deterministic=False, repeat=None):
        samples, log_prob = super(HierarchicalPolicy, self).__call__(
            observations=observations,
            deterministic=deterministic,
            repeat=repeat,
        )
        samples = samples * (self.coefficient_range[1] - self.coefficient_range[0]) + self.coefficient_range[0]
        return samples, log_prob


class FamilyTanhGaussianPolicy(TanhGaussianPolicy):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0

    def setup(self):
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim, arch=self.arch, orthogonal_init=self.orthogonal_init
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        return action_distribution.log_prob(actions)

    def __call__(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=self.make_rng('noise'))

        return samples, log_prob
