# FamO2O's implementation over Implicit Q-Learning (IQL)

This repository provides FamO2O's implementation over implicit Q-learning (IQL).  The results of FamO2O+IQL on D4RL AntMaze can be reproduced by these codes.

## How to run the code

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

```bash
python family_train_finetune.py \
    --env_name=${ENV} \
    --config=configs/family_antmaze_finetune_config.py \
    --eval_episodes=100 \
    --eval_interval=100000 \
    --last_eval_interval=10000 \
    --last_eval_start_steps=60000 \
    --replay_buffer_size 2000000 \
    --seed ${SEED} \
    --save_dir logs/family_iql_finetune/family_antmaze_finetune_config/${ENV}/seed_${SEED}
```

Here, `${ENV}` can be one of the following environments:

- `antmaze-umaze-v0`
- `antmaze-umaze-diverse-v0`
- `antmaze-medium-v0`
- `antmaze-medium-diverse-v0`
- `antmaze-large-v0`
- `antmae-large-diverse-v0`

`${SEED}` refers to the random seed. In our paper, to avoid cherry-picking random seeds, we use `0, 1, 2, 3, 4, 5` for 6 repeated experiments.

## Acknowledgment

The implementation is based on [IQL's codes](https://github.com/ikostrikov/implicit_q_learning).
