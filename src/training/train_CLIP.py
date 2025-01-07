"""
Train the conditional denoising tasks.
"""
import os 
import sys
import torch 
import numpy as np 
import time 
import wandb 
from transformers import HfArgumentParser

src_path = os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(src_path)
model_path = os.path.join(os.path.dirname(__file__), '..','models')
sys.path.append(model_path)

from src.data import *
from src.models import *
from src.utils import * 

# Generate the Config class for the training script
@dataclass 
class TrainingConfig(UtilConfig, DoubleTreeConfig, ClipModelConfig):
    job_name: Optional[str] = field(default='clip')

parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)
guide = clip_guide
# CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Model Input Dimension 
d_tmodel = n_ttree_child**n_ttree_layer
d_imodel = n_itree_child**n_itree_layer
d_model =d_imodel + d_tmodel
# Model 
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
tree_folder = f'K{K}_L{n_ttree_layer}C{n_ttree_child}p{int(p_ttree_flip*100)}_L{n_itree_layer}C{n_itree_child}p{int(p_itree_flip*100)}sc{int(flip_scale*10)}'
model_name = f'L{clip_tmodel_nlayer}H{clip_tmodel_nhead}D{clip_tmodel_deb}_L{clip_imodel_nlayer}H{clip_imodel_nhead}D{clip_imodel_deb}'
tags=[job_name, tree_folder]
if guide: 
    tags.append('guide')
    model_name = 'GT_' + model_name
else: 
    model_name = 'TF_' + model_name

# Initialize Loggers 
directory = os.path.join("./logs", job_name, tree_folder, model_name, timestamp) 
logger = GenLogger(directory, config, raw=raw)
if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

# sampler 

p_y = np.ones(variable_type) / variable_type 
sampler = ClipSampler([n_ttree_layer, n_itree_layer], 
                    [n_ttree_child, n_itree_child], 
                    [p_y, p_y], 
                    [p_ttree_flip, p_itree_flip], 
                    K=K,
                    flip_scale=flip_scale,
                    variable_type=variable_type, 
                    translation_invariance=True, 
                    seedtree=42)
Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
logger.info(f'Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}')
if not raw:
    wandb.log({'Bayes_loss': Bayes_loss, 'Bayes_std': Bayes_std})

# Model 
seed_everything(seed)
tmodel = EncoderTransformer(n_token=d_tmodel, 
                           num_class=variable_type, 
                           n_embd=clip_tmodel_deb, 
                           n_layer=clip_tmodel_nlayer, 
                           n_guided_layer=n_ttree_layer,
                           n_head=clip_tmodel_nhead, 
                           n_mlp_multiplier=4, 
                           activation=clip_activation,  
                           mlp=True, 
                           normalize_attn=clip_attennorm, 
                           layernorm=clip_layernorm, 
                           guide=clip_guide) # text model

imodel = EncoderTransformer(n_token=d_imodel,
                            num_class=variable_type,
                            n_embd=clip_imodel_deb,
                            n_layer=clip_imodel_nlayer,
                            n_guided_layer=n_itree_layer,
                            n_head=clip_imodel_nhead,
                            n_mlp_multiplier=4,
                            activation=clip_activation,
                            mlp=True,
                            normalize_attn=clip_attennorm,
                            layernorm=clip_layernorm,
                            guide=clip_guide) # image model
tmodel = tmodel.to(device)
imodel = imodel.to(device)

# Loss and optimizer
loss = GuidedClipLoss(K, batch_size, penalty=penalty, guide=guide)
loss_nop = GuidedClipLoss(K, batch_size, penalty=0, guide=False)

optimizer = AdamW(params=list(tmodel.parameters()) 
					   + list(imodel.parameters()), lr=None) # optimizer

ploss_history = np.zeros(total_iters+1)
loss_history = np.zeros(total_iters+1)
# compare_history = np.zeros(total_iters)

# loading from checkpoint
if init_from != 'scratch':
    ckpt_dir = checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    tmodel.load_state_dict(checkpoint['tmodel_state_dict']) # load the state dict for text model
    imodel.load_state_dict(checkpoint['imodel_state_dict']) # load the state dict for image model
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # load the state dict for the optimizer
    iter_num = checkpoint['iter'] # load the iteration number


# Main Loop
curr_time = time.time() 
iter_num = 0
# total_iter = 1 
while iter_num < total_iters+1: 
    
    optimizer.zero_grad() 

    res_text, res_image = sampler.get_batch(device=device, batch_size=batch_size, guide=guide)
    guided_layers = [res_text[2], res_image[2]]
    
    tmodel_output = tmodel(res_text[0])
    imodel_output = imodel(res_image[0])

    output = loss(tmodel_output, imodel_output, guided_layers)
    output_nop = loss_nop(tmodel_output, imodel_output, guided_layers)

    output[0].backward() 

    ploss_history[iter_num] = output[0].item()
    loss_history[iter_num] = output_nop[0].item()

    torch.nn.utils.clip_grad_norm_(list(tmodel.parameters()) 
					   + list(imodel.parameters()), max_norm, norm_type=2) # clip the gradient
    lr = get_lr_cosine_schedule(iter_num, lr_max, lr_min, warmup_iters, total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    
    finish_time = time.time()

    if iter_num>0 and iter_num % log_interval == 0:
        logger.info((
            f'Iter: {iter_num}, '
            f'Penalty train loss: {np.mean(ploss_history[iter_num//2:iter_num]):.4f}, '
            f'Train loss: {np.mean(loss_history[iter_num//2:iter_num]):.4f}, '
            f'Guided penalty: [{output[1]:.4f}],'
            f'Bayes: {Bayes_loss:.4f}, '
            f'LR: {lr:.6f}, '
            f'Time: {(finish_time - curr_time):.2f}s'
        ))
        if not raw:
            wandb.log({'train_loss': loss_history[iter_num],
                        'penalty_train_loss':ploss_history[iter_num], 
                        'lr': lr, 
                        'Bayes_loss': Bayes_loss, 
                        'Bayes_std': Bayes_std, 
                        'iter': iter_num})
                        
    
    if iter_num % eval_interval == 0 and not raw:
        torch.save({'tmodel_state_dict': tmodel.state_dict(),
                    'imodel_state_dict':imodel.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss, 
                    'iter': iter_num, 
                    'loss_history':loss_history, 
                    'ploss_history':ploss_history, 
                    'bayes':Bayes_loss}, 
                    checkpoint_path)
    iter_num += 1

if not raw:
    torch.save({'tmodel_state_dict': tmodel.state_dict(),
                'imodel_state_dict':imodel.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'loss': loss, 
                'iter': iter_num, 
                'loss_history':loss_history, 
                'ploss_history':ploss_history, 
                'bayes':Bayes_loss}, 
                checkpoint_path)
    
logging.shutdown() # close the logger

if S3_upload: 
    import s3fs
    s3_file = s3fs.S3FileSystem()
    local_path = directory
    s3_path = S3_bucket_name+f'/GHM/{job_name}/{tree_folder}/{model_name}/{timestamp}'
    s3_file.put(local_path, s3_path, recursive=True) 
