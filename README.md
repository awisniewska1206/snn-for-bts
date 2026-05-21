# Brain-to-Text

This repository contains code for "Efficient Brain-to-Speech Decoding with Fixed-Delay Spiking Neural Networks"

Right now the repo is organized around three main pieces:

- `model_training/` for training, evaluation, configs, and model implementations
- `analyses/` for notebooks, summaries, and generated figures

## Setup

Create the environment from the provided Conda file:

```bash
conda env create -f environment.yml
conda activate spikachu
```

## Status

This README is in its draft version as of now.

python model_training/run_family_experiments_daywise_splits.py   --model snn   --gpu 0   --seed 20   --split_dir data/daywise_sentence_splits --split_iterations 0

python model_training/optimize_snn_hparams.py --model_type snn --gpu 0

python model_training/helpers/run_daywise_eval_all.py   --runs_dir model_training/logs/model_type   --gpu_number 0

python model_training/helpers/evaluate_daywise_test_split.py --model_path model_training/logs/model_type/run_dir --split_number 0 --gpu_number 0