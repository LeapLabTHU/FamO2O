import os

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    config.opt_decay_schedule = None  # Don't decay optimizer lr

    # add for FamO2O
    config.config_name = os.path.basename(__file__).split('.')[0]
    config.family_sin_cos_n = 10000.0
    config.family_sin_cos_d = 6
    config.family_coefficient_range = (8.0, 14.0)
    config.target_entropy = None
    config.hierarchical_temp_lr = 3e-4
    config.hierarchical_init_temperature = 1.0

    return config
