# Multimodal Learning with GHM

This repository provides the official implementation for the paper: 

**A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI.**

Oko Kazusato, Licong Lin, Yuhang Cai, Song Mei. 

Paper: add link here. 

## Setup 

To get started, create a new conda environment and install all required packages:

```shell
conda create -n GHM python=3.10 --yes
conda activate GHM
pip install -r requirements.txt
```

### Repository Structure

- `experiments/`
  Contains shell scripts for running experiments and illustrative examples.
- `saved_models/`
  Houses two pretrained CLIP models: one using ReLU attention and another using softmax attention.
- `src/`
  Source code directory:
  - `src/data/data_random_GHM.py`
    Generates random data using various samplers.
  - `src/models/model.py`: 
    Defines models for all tasks.
  - `src/models/optimizer.py`
    Implements optimization algorithms.
  - `src/training/train_[task_name].py`
    Specifies training routines for individual tasks.
  - `src/utils/`
    Contains utility scripts for configuration initialization and logging.
- `logs`
  Contains log files for all trainings.

## Training

To train a model for a specific task, navigate to the `experiments/` directory and run:

```
bashCopy codechmod +x ./eg_[task_name].sh
./eg_[task_name].sh
```

Here, `[task_name]` can be one of:

- `clip`: CLIP task.
- `dns`: Jointly trains a CLIP model and a denoising model for conditional denoising tasks.
- `sdns`: Trains only the denoising model while keeping the CLIP features fixed for conditional denoising tasks.
- `nwp`: Jointly trains a CLIP model and a transformer for next-word prediction.
- `snwp`: Trains only the transformer while keeping the CLIP features fixed for next-word prediction.

For more details about each configuration, refer to:

- `src/training/train_[task_name].py`
- `src/utils/config.py`

You can modify the provided shell scripts in `experiments/` to change training parameters as needed.

## Miscellaneous

If you find this code useful in your research, please cite our paper:

```


```

Thank you for your interest in our work! If you have any questions or encounter any issues, feel free to open an issue or submit a pull request.

