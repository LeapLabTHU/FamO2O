# [NeurIPS 2023 Spotlight] Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning

![image](teaser.svg)

This repository is the official source code for "Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning" [[project page]](https://shenzhi-wang.github.io/NIPS_FamO2O/) [[NeurIPS page]](https://openreview.net/forum?id=vtoY8qJjTR&referrer=[Author Console](%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions)), which has been accepted as a **spotlight** presentation at NeurIPS 2023.

This codebase includes:

1. The implementation of FamO2O using JAX IQL, located in the [jax_iql folder](jax_iql/). For detailed instructions, please see the [jax_iql README](jax_iql/README.md).
2. The implementation of FamO2O using JAX CQL, located in the [jax_cql folder](jax_cql/). For additional information, please refer to the [jax_cql README](jax_cql/README.md).

We would greatly appreciate it if you could cite our work!

```
@inproceedings{
wang2023train,
title={Single Training Session, Multiple Model Outputs: Dynamic Balancing for Offline-to-Online Reinforcement Learning},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=vtoY8qJjTR}
}
```