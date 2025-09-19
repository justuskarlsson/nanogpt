# Modded NanoGPT - Refactored

This repository is a fork of the [modded-nanogpt codebase](https://github.com/KellerJordan/modded-nanogpt), split into modular components for easier development and different use cases.
Changes:
- Made to work A100 gpus (instead of H100)
- Refactored code, split into multiple files
- Trained on Medium instead of Small model
- [WIP] Also do finetuning and some evals

Managed to pre-train the Medium model from scratch to 70k epochs. Results (complete the sentence):


## Training
conda activate modded
* sh run.sh
* sh finetune.sh
* python evaluation_benchmarks.py

## Medium 70k epoch (temp 0.7)
