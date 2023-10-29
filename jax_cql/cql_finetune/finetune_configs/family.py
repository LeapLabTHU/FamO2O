from copy import deepcopy

from cql_finetune.finetune_configs.base import config as base_config

config = deepcopy(base_config)
config['coefficient_range'] = (0.5, 1.5)
config["cql"].cql_min_q_weight = 1.0
config["cql"].hierarchical_policy_lr = 3e-4
config["cql"].sin_cos_n = 10000.0
config["cql"].sin_cos_d = 6
