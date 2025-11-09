# Multimodal Learning with GHM

This repository contains the official implementation of the paper:

**A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI**  
Oko Kazusato, Licong Lin, Yuhang Cai, Song Mei  
Paper: https://arxiv.org/abs/2501.04641 

## Setup 

To get started, create a new `uv` environment (Python 3.10) and install the package in editable mode:

```shell
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Repository Structure

- `scripts/`  
  Shell scripts for running experiments and illustrative examples:
  - `scripts/examples/`: Example scripts for individual tasks (`eg_clip.sh`, `eg_dns.sh`, `eg_sdns.sh`, `eg_nwp.sh`, `eg_snwp.sh`, `eg_clas.sh`)
  - `scripts/experiments/`: Batch experiment scripts for different model configurations (`exp_clip_*.sh`, `exp_cdm_*.sh`, `exp_vlm_*.sh`)
- `saved_models/`  
  Pretrained CLIP models (softmax attention variants available)
- `src/ghmclip/`  
  Main source code package:
  - `src/ghmclip/data/data_random_GHM.py`: Generates random data using various samplers
  - `src/ghmclip/models/model.py`: Defines models for all tasks
  - `src/ghmclip/models/optimizer.py`: Implements optimization algorithms
  - `src/ghmclip/training/`: Training routines for different tasks:
    - `train_CLIP.py`: CLIP contrastive learning
    - `train_CDNS.py`: Conditional denoising (joint training)
    - `train_sequential_DNS.py`: Sequential denoising (fixed CLIP features)
    - `train_NWP.py`: Next-word prediction (joint training)
    - `train_sequential_NWP.py`: Sequential next-word prediction (fixed CLIP features)
    - `train_CLS.py`: Classification tasks
  - `src/ghmclip/utils/`: Utility scripts for configuration and logging
- `figures/` Figure-generation scripts and rendered figures
- `logs/` Log files organized by task type (CDM, VLM) and experiment configurations
- `tests/` Unit tests
- `main.py` Main entry point

## Checkpoints 
All checkpoints are available on Hugging Face: https://huggingface.co/faro1219/multimodal-ghm.  
Download them in advance if you plan to reproduce the figures. Download the folder `logs/CLIP` in advance, 
if you want to run those training scripts with `sequential`. 

## Training

To train a model for a specific task, navigate to the `scripts/examples/` directory and run:

```bash
chmod +x ./eg_[task_name].sh
./eg_[task_name].sh
```

Here, `[task_name]` can be one of:

- `clip`: CLIP contrastive learning task
- `dns`: Conditional denoising (jointly trains CLIP and denoising model)
- `sdns`: Sequential denoising (trains only denoising model with fixed CLIP features)
- `nwp`: Next-word prediction (jointly trains CLIP and transformer)
- `snwp`: Sequential next-word prediction (trains only transformer with fixed CLIP features)
- `clas`: Classification tasks

For batch experiments with different configurations, use the scripts in `scripts/experiments/`:

- `exp_clip_*.sh`: CLIP experiments with different model architectures
- `exp_cdm_*.sh`: Conditional denoising experiments
- `exp_vlm_*.sh`: Vision–language model experiments

For more details about each configuration, refer to:

- `src/ghmclip/training/train_[task_name].py`
- `src/ghmclip/utils/config.py`

You can modify the provided shell scripts in `scripts/examples/` and `scripts/experiments/` to change training parameters as needed.

## Citation

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