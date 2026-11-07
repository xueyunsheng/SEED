# Decoupling is the Key: Scaling Deep Value Networks in Reinforcement Learning-SEED

This repository contains a cleaned **online SAC implementation** of SEED for continuous-control Gymnasium environments. It combines:

1. A bilinear Koopman autoencoder pretrained on environment transitions. The action-independent drift matrix provides slow spectral modes; a frozen encoder projects observations onto them.
2. A deep residual categorical critic. Koopman features are fused at the final value head, giving the deep backbone a separate path for residual features.
3. An adaptive symmetric value support. A truncated Gaussian target is matched to the scalar SAC Bellman target and the support radius is optimized using projection and confidence losses.

The training entrypoint is [main.py](main.py), with the algorithm in [agents/seed.py](agents/seed.py) and supporting modules in `utils/`. This is a JAX reimplementation of the paper.

## Repository layout

```text
main.py                 Online training and CLI
agents/seed.py          SAC actor, categorical critic, and update step
utils/koopman.py        Bilinear dynamics, pretraining, spectral basis
utils/distribution.py   Adaptive value support and target distribution
utils/replay_buffer.py  Replay buffer
requirements.txt        Python dependencies
```

## Install

Use Python 3.10+ and install the JAX build appropriate for your platform, then:

```bash
python -m pip install -r requirements.txt
```

For CUDA, follow the official JAX installation instructions for the appropriate CUDA version. W&B logging requires an additional `wandb` install and `--track`.

## Koopman pretraining data

Before online training, provide an `.npz` file at `datasets/<ENV_ID>_step_<N>.npz`, or pass `--dataset-dir` and `--offline-dataset-steps`. Required arrays are `observations`, `actions`, and `next_observations`, each with the same nonzero number of transitions, in the environment's original observation and action units. For example, the defaults expect `datasets/Hopper-v4_step_100000.npz`.

The paper's online protocol uses **500,000 transitions from a standard SAC warmup policy** for Koopman pretraining. Match that protocol with a dataset named `Hopper-v4_step_500000.npz` and pass `--offline-dataset-steps 500000`. The warmup collector and dataset are not present in the supplied source tree; supplying arbitrary transitions changes the experiment. Observations are normalized using the dataset mean and standard deviation, and the same normalization is applied online.

## Run

```bash
python main.py --env-id Hopper-v4 --dataset-dir datasets --offline-dataset-steps 500000 --koopman-train-steps 30000 --total-timesteps 1000000
```

`--num-blocks` controls critic depth. Each residual block contains two dense layers plus the input and output dense layers. With the default two blocks, the critic has six dense layers. TensorBoard logs go to `runs/`.

## Scope and implementation notes

- The critic predicts categorical logits and uses cross entropy against adaptive truncated Gaussian targets. The actor uses the expectation of this distribution.
- The Koopman encoder is frozen after pretraining. Eigenvectors are selected by eigenvalue distance to the unit circle; real and imaginary parts are concatenated as features.
- The online loop currently supports one environment (`--num-envs 1`).
