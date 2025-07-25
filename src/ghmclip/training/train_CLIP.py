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

from ghmclip.data import *
from ghmclip.models import *
from ghmclip.utils import * 

# Generate the Config class for the training script
@dataclass 
class TrainingConfig(UtilConfig, DoubleTreeConfig, ClipModelConfig):
    job_name: Optional[str] = field(default='CLIP')
    exp_uid: Optional[str] = field(default='')
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)
guide = clip_guide
# CUDA device

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
    tags.append('GT')
    model_name = 'GT_' + model_name
else: 
    model_name = 'TF_' + model_name
    if clip_tmodel_nlayer == 1:
        tags.append('1T')
    else:
        tags.append('ST')

# Initialize Loggers 
directory = os.path.join("./logs", job_name, tree_folder, model_name, timestamp) 
logger = GenLogger(directory, config, raw=raw)
if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

# sampler 

p_y = torch.ones(variable_type) / variable_type 
# sampler = ClipSampler([n_ttree_layer, n_itree_layer], 
                    # [n_ttree_child, n_itree_child], 
                    # [p_y, p_y], 
                    # [p_ttree_flip, p_itree_flip], 
                    # K=K,
                    # flip_scale=flip_scale,
                    # variable_type=variable_type, 
                    # translation_invariance=True, 
                    # seedtree=seed_tree,
                    # device=device)

sampler = DoubleSampler(n_layers=[n_ttree_layer, n_itree_layer], 
                        n_childs=[n_ttree_child, n_itree_child], 
                        p_ys=[p_y, p_y],
                        p_flips=[p_ttree_flip, p_itree_flip], 
                        flip_scale=flip_scale,
                        variable_type=variable_type, 
                        translation_invariance=True, 
                        seedtrees=[seed_ttree, seed_itree],
                        device=device)

Bayes_loss, Bayes_std = sampler.get_bayes(n_eval=12800, job='clip', batch_size=batch_size)
logger.info(f'Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}')
if not raw:
    wandb.log({'Bayes_loss': Bayes_loss, 'Bayes_std': Bayes_std})

# Model 
guide_dim = gen_guide_indices(job='clip', n_tree_layer=n_ttree_layer, n_class=variable_type)

tmodel = EncoderTransformer(n_head=clip_tmodel_nhead,
                            d_model=clip_tmodel_deb,
                            d_ff=clip_tmodel_deb*4,
                            n_layer=clip_tmodel_nlayer,
                            guide=clip_guide,
                            n_class=variable_type,
                            n_token=d_tmodel, 
                            job='clip',
                            guide_dim=guide_dim,
                            auto_regressive=False,
                            ) # text model

imodel = EncoderTransformer(n_head=clip_imodel_nhead,
                            d_model=clip_imodel_deb,
                            d_ff=clip_imodel_deb*4,
                            n_layer=clip_imodel_nlayer,
                            guide=clip_guide,
                            n_class=variable_type,
                            n_token=d_imodel, 
                            job='clip',
                            guide_dim=guide_dim, 
                            auto_regressive=False) # image model
tmodel = tmodel.to(device)
imodel = imodel.to(device)

# Loss and optimizer
loss = CLIPLoss(penalty=penalty, normalize=clip_normalize, temperature=clip_temperature)
loss = loss.to(device)  

# loss = nn.CrossEntropyLoss()

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

    res_text, res_image = sampler.get_batch(batch_size=batch_size, guide=guide)
    guided_layers = [res_text[2], res_image[2]]
    
    tmodel_output, tmodel_guided_input, _ = tmodel(res_text[0])
    imodel_output, imodel_guided_input, _ = imodel(res_image[0])

    loss_input = [[tmodel_output, tmodel_guided_input], [imodel_output, imodel_guided_input]]
    output, loss_val, penality_term = loss(loss_input, guided_layers, guide=clip_guide)

    output.backward() 

    ploss_history[iter_num] = penality_term
    loss_history[iter_num] = loss_val

    torch.nn.utils.clip_grad_norm_(list(tmodel.parameters()) 
					   + list(imodel.parameters()), max_norm, norm_type=2) # clip the gradient
    lr = get_lr_cosine_schedule(iter_num, lr_max, lr_min, warmup_iters, total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    
    finish_time = time.time()

    if iter_num>0 and iter_num % log_interval == 0:
        logger.info((
            f'Iter: {iter_num}, '
            f'Ploss: {output.item():.4f}, '
            f'Loss: {loss_val:.4f}, '
            f'GuideP: {penality_term:.4f}, '
            f'Bayes: {Bayes_loss:.4f}, '
            f'LR: {lr:.6f}, '
            f'Time: {(finish_time - curr_time):.2f}s'
        ))
        if not raw:
            wandb.log({'train_loss': loss_val,
                        'penalty_train_loss':output.item(),
                        'guided_penalty':penality_term,
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
