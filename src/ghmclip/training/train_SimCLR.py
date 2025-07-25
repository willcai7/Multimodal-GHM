"""
Train the next word prediction tasks.
"""
import os 
import sys
import torch 
import numpy as np 
import time 
import wandb 
from transformers import HfArgumentParser
from ghmclip.data import SingleSampler, Augmentator
from ghmclip.models import *
from ghmclip.utils import * 
from ghmclip.models.functional import *

# Generate the Config class for the training script
@dataclass 
class TrainingConfig(UtilConfig, TreeConfig, ModelConfig):
    job_name: Optional[str] = field(default='SimCLR')
    temperature: Optional[float] = field(default=1.0)
    loss_norm: Optional[bool] = field(default=True)
    raw: Optional[bool] = field(default=True)

parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

# CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Model Input Dimension 
d_model = n_tree_child**n_tree_layer

# Model 
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
tree_folder = f'K{K}_L{n_tree_layer}C{n_tree_child}p{int(p_tree_flip*100)}_sc{int(flip_scale*10)}'
model_name = f'L{n_model_layer}H{n_head}D{d_eb}'
tags=[job_name, tree_folder]
model_name = 'TF_' + model_name

# Initialize Loggers 
directory = os.path.join("./logs", job_name, tree_folder, model_name, timestamp) 
logger = GenLogger(directory, config, raw=raw)
if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

# sampler 

p_y = torch.ones(variable_type) / variable_type 
sampler = SingleSampler(
        n_layer=n_tree_layer,
        n_child=n_tree_child,
        p_y=p_y,
        p_flip=p_tree_flip,
        flip_scale=flip_scale,
        variable_type=variable_type,
        seedtree=seed_tree,
        sigma=1,
        device=device,
    )
augmentator = Augmentator()

# Model 
seed_everything(seed)
model = EncoderTransformer(n_token=d_model,
                           n_class=variable_type,
                           n_layer=n_model_layer, 
                           d_model=d_eb,
                           d_ff=d_eb*4, 
                           n_head=n_head,
                           job='clip')
model = model.to(device)
loss =  CLIPLoss(temperature=temperature, normalize=loss_norm)
optimizer = AdamW(params=model.parameters(), lr=None)
loss_history = np.zeros(total_iters)

# loading from checkpoint
if init_from != 'scratch':
    ckpt_dir = checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter']


# Main Loop
curr_time = time.time() 
iter_num = 0
total_iter = 1 
while iter_num < total_iters: 
    optimizer.zero_grad() 
    images, roots,_,_ = sampler.get_batch(job='clas', batch_size=batch_size)
    images_aug = augmentator(images)
    out,_,_ = model(images) 
    out_aug,_,_ = model(images_aug)
    # print(out.shape, out_aug.shape)
    # print(out.shape, out_aug.shape)
    loss_input = [[out, roots], [out_aug, roots]]

    output,loss_val,_ = loss(loss_input, guide=False, targets=None)
    output.backward() 

    loss_history[iter_num] = loss_val
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
    lr = get_lr_cosine_schedule(iter_num, lr_max, lr_min, warmup_iters, total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    if iter_num>0 and iter_num % log_interval == 0:
        logger.info(
            f'Iter: {iter_num}, '
            f'Train loss: {np.mean(loss_history[iter_num//2:iter_num]):.4f}, '
            f'LR: {lr:.6f}, '
            f'Time: {(finish_time - curr_time):.2f}s'
        )
# Compare: {np.mean(compare_history[iter_num//2: iter_num]):.4f},
        if not raw:
            wandb.log({'train_loss': loss_history[iter_num],
                        'lr': lr, 
                        'iter': iter_num})
                        
    
    if iter_num % eval_interval == 0 and not raw:
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                'loss': loss, 'iter': iter_num, 'loss_history':loss_history}, checkpoint_path)
    iter_num += 1

logging.shutdown() # close the logger

if not raw:
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
            'loss': loss, 'iter': iter_num, 'loss_history':loss_history}, checkpoint_path) 
       
if S3_upload: 
    import s3fs
    s3_file = s3fs.S3FileSystem()
    local_path = directory
    s3_path = S3_bucket_name+f'/GHM/{job_name}/{tree_folder}/{model_name}/{timestamp}'
    s3_file.put(local_path, s3_path, recursive=True) 