# FamO2O's implementation over Conservative Q-Learning (CQL)

This repository provides FamO2O's implementation over conservative Q-learning (CQL).  The results of FamO2O+CQL on D4RL Locomotion can be reproduced by these codes.

# How to run the code

## Install dependencies

1. Install and use the included Anaconda environment

```
$ conda env create -f environment.yml
$ source activate JaxCQL
```

You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable.

```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run training

1. If you do not want sync with wandb, please use the following command:

```bash
python -m cql_finetune.family_conservative_sac_finetune_main --env ${ENV} --seed ${SEED} -c family --wandb-offline --wandb-output-dir "./experiment_output"
```

Here, `${SEED}` refers to the random seed. In our paper, to avoid cherry-picking random seeds, we use `0, 1, 2, 3, 4, 5` for 6 repeated experiments.

`${ENV}` can be one of the following environments:

- `halfcheetah-medium-v2`
- `halfcheetah-medium-replay-v2`
- `halfcheetah-medium-expert-v2`
- `walker2d-medium-v2`
- `walker2d-medium-replay-v2`
- `walker2d-medium-expert-v2`
- `hopper-medium-v2`
- `hopper-medium-replay-v2`
- `hopper-medium-expert-v2`

2. If you need to sync with wandb, please use the following command:

```bash
python -m cql_finetune.family_conservative_sac_finetune_main --env ${ENV} --seed ${SEED} -c family --wandb-output-dir "./experiment_output" --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}
```

Here, the `${WANDB_PROJECT}` and `${WANDB_ENTITY}` should be substituted by your own wandb project and entity, respectively.

## Acknowledgment

The implementation is based on [CQL](https://github.com/young-geng/JaxCQL).
