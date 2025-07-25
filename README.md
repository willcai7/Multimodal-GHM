# Multimodal Learning with GHM

This repository provides the official implementation for the paper: 

**A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI.**

Oko Kazusato, Licong Lin, Yuhang Cai, Song Mei. 

Paper: https://arxiv.org/abs/2501.04641 

## Setup 

To get started, create a new `uv` environment and install all required packages:

```shell
uv venv --python 3.12
source activate .venv/bin/activate
uv pip install -e .
```

### Repository Structure

- `experiments/`
  Contains shell scripts for running experiments and illustrative examples.
- `saved_models/`
  Houses two pretrained CLIP models: one using ReLU attention and another using softmax attention.
- `src/ghmclip/`
  Source code directory:
  - `src/ghmclip/data/data_random_GHM.py`
    Generates random data using various samplers.
  - `src/ghmclip/models/model.py`: 
    Defines models for all tasks.
  - `src/ghmclip/models/optimizer.py`
    Implements optimization algorithms.
  - `src/ghmclip/training/train_[task_name].py`
    Specifies training routines for individual tasks.
  - `src/ghmclip/utils/`
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

- `CLS`: CLIP task.
- `CDNS`: Jointly trains a CLIP model and a denoising model for conditional denoising tasks.
- `sequential_DNS`: Trains only the denoising model while keeping the CLIP features fixed for conditional denoising tasks.
- `NWP`: Jointly trains a CLIP model and a transformer for next-word prediction.
- `sequential_NWP`: Trains only the transformer while keeping the CLIP features fixed for next-word prediction.
- `SimCLR`: Trains a SimCLR model.

For more details about each configuration, refer to:

- `src/ghmclip/training/train_[task_name].py`
- `src/ghmclip/utils/config.py`

You can modify the provided shell scripts in `experiments/` to change training parameters as needed.

## Miscellaneous

If you find this code useful in your research, please cite our paper:

```
@article{oko2025multimodal,
      title={A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI}, 
      author={Kazusato Oko and Licong Lin and Yuhang Cai and Song Mei},
      year={2025},
      archivePrefix={arXiv},
}
```

Thank you for your interest in our work! If you have any questions or encounter any issues, feel free to open an issue or submit a pull request.

