import hashlib
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np


def encode(name: str, length=64):
    m = hashlib.md5()
    m.update(str.encode(name))
    return m.hexdigest()[:length]


@partial(jax.jit, static_argnames=('n', 'd'))
def sin_cos_skill_func(
        beta_array: Union[np.ndarray, jnp.ndarray],
        n: float = 10000.0,
        d: int = 6,
):
    k = beta_array.reshape(-1, 1)
    if isinstance(beta_array, np.ndarray):
        arange_func = np.arange
        sin_func = np.sin
        cos_func = np.cos
        empty_func = np.empty
    elif isinstance(beta_array, jnp.ndarray):
        arange_func = jnp.arange
        sin_func = jnp.sin
        cos_func = jnp.cos
        empty_func = jnp.empty
    else:
        raise TypeError(f'Invalid type of beta_array: {type(beta_array)}')

    i = arange_func(0, d / 2, 1)

    inner_value = k / (n ** (2 * i / d))
    skill = empty_func((k.shape[0], d))
    if isinstance(beta_array, np.ndarray):
        skill[:, 0:d:2] = sin_func(inner_value)
        skill[:, 1:d:2] = cos_func(inner_value)
    elif isinstance(beta_array, jnp.ndarray):
        skill = skill.at[:, 0:d:2].set(sin_func(inner_value))
        skill = skill.at[:, 1:d:2].set(cos_func(inner_value))
    else:
        raise TypeError(f'Invalid type of beta_array: {type(beta_array)}')

    return skill
