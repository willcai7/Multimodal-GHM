from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable

@dataclass 
class TreeConfig:
    n_tree_layer: Optional[int] = field(default=3)
    n_tree_child: Optional[int] = field(default=4)
    p_tree_flip: Optional[float] = field(default=0.10)
    sigma: Optional[float] = field(default=1)
    K: Optional[int] = field(default=4)
    batch_size: Optional[int] = field(default=128)
    variable_type: Optional[int] = field(default=10)
    flip_scale: Optional[float] = field(default=1.0)
    translation_invariance: Optional[bool] = field(default=True)

@dataclass
class DoubleTreeConfig:
    n_ttree_layer: Optional[int] = field(default=3)
    n_itree_layer: Optional[int] = field(default=3)
    n_ttree_child: Optional[int] = field(default=4)
    n_itree_child: Optional[int] = field(default=4)
    p_ttree_flip: Optional[float] = field(default=0.10)
    p_itree_flip: Optional[float] = field(default=0.10)
    sigma: Optional[float] = field(default=1)
    K: Optional[int] = field(default=4)
    batch_size: Optional[int] = field(default=128)
    variable_type: Optional[int] = field(default=10)
    flip_scale: Optional[float] = field(default=1.0)
    translation_invariance: Optional[bool] = field(default=True)

@dataclass 
class ModelConfig:
    model_type: Optional[str] = field(default='TF')
    n_model_layer: Optional[int] = field(default=10)
    d_eb: Optional[int] = field(default=64)
    n_head: Optional[int] = field(default=4)
    residual_pdrop: Optional[float] = field(default=0.0)
    layernorm: Optional[bool] = field(default=False)
    normalize_attn: Optional[bool] = field(default=True)
    guide: Optional[bool] = field(default=False)
    activation: Optional[str] = field(default='softmax')

@dataclass 
class ClipModelConfig:
    clip_model_type: Optional[str] = field(default='TF')
    clip_tmodel_nlayer: Optional[int] = field(default=10)
    clip_imodel_nlayer: Optional[int] = field(default=10)
    clip_tmodel_deb: Optional[int] = field(default=64)
    clip_imodel_deb: Optional[int] = field(default=64)
    clip_tmodel_nhead: Optional[int] = field(default=4)
    clip_imodel_nhead: Optional[int] = field(default=4)
    clip_residual_pdrop: Optional[float] = field(default=0.0)
    clip_layernorm: Optional[bool] = field(default=False)
    clip_attennorm: Optional[bool] = field(default=True)
    clip_guide: Optional[bool] = field(default=False)
    clip_activation: Optional[str] = field(default='softmax')

@dataclass
class OptimizerConfig:
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=5e-6)
    warmup_iters: Optional[int] = field(default=0)
    total_iters: Optional[int] = field(default=2*(10**4))
    max_norm: Optional[float] = field(default=1.0)
    penalty: Optional[float] = field(default=0.001)
    loss_type: Optional[str] = field(default="exp")
    init_from: str = 'scratch'

@dataclass
class LoggingConfig:
    log_interval: Optional[int] = field(default=20)
    eval_interval: Optional[int] = field(default=200)
    eval_iters: Optional[int] = field(default=1200)
    wandb_logging: bool = True
    wandb_project: str = "Clip-GHM"
    wandb_path: str = "./others/wandb"
    raw: Optional[bool] = field(default=True)
    seed: Optional[int] = field(default=224)
    S3_upload: Optional[bool] = field(default=False)
    S3_bucket_name: Optional[str] = field(default='yuhangbucket')

@dataclass 
class UtilConfig(LoggingConfig, OptimizerConfig):
    device: Optional[str] = field(default='cuda')

