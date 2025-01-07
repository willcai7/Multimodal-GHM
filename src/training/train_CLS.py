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

src_path = os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(src_path)
model_path = os.path.join(os.path.dirname(__file__), '..','models')
sys.path.append(model_path)

from src.data import *
from src.models import *
from src.utils import * 

# Generate the Config class for the training script
@dataclass 
class TrainingConfig(UtilConfig, TreeConfig, ModelConfig):
    job_name: Optional[str] = field(default='classification')

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
tree_folder = f'L{n_tree_layer}C{n_tree_child}p{int(p_tree_flip*100)}'
model_name = f'L{n_model_layer}H{n_head}D{d_eb}'
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
# sampler = ClassificationSampler(n_layer=n_tree_layer, 
#                                 n_child=n_tree_child,
#                                 p_y=p_y, 
#                                 p_flip=p_tree_flip, 
#                                 variable_type=variable_type, 
#                                 flip_scale=flip_scale, 
#                                 translation_invariance=translation_invariance, 
#                                 )


n_ttree_layer = 4
n_itree_layer = 4
n_ttree_child = 3
n_itree_child = 3
p_ttree_flip = 0.4
p_itree_flip = 0.4
flip_scale = 1
variable_type = 10

sampler = NextWordPredictSampler([n_ttree_layer, n_itree_layer], 
                                [n_ttree_child, n_itree_child], 
                                [p_y, p_y], 
                                [p_ttree_flip, p_itree_flip], 
                                flip_scale=flip_scale,
                                variable_type=variable_type, 
                                translation_invariance=True, 
                                seedtree=42)

Bayes_loss = 0.1
Bayes_std = 0.1
# Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
# logger.info(f'Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}')
# if not raw:
#     wandb.log({'Bayes_loss': Bayes_loss, 'Bayes_std': Bayes_std})

# Model 
seed_everything(seed)
model = EncoderTransformer(n_token=d_model,
                           num_class=variable_type,
                           n_layer=n_model_layer, 
                           n_embd=d_eb, 
                           n_head=n_head, 
                           guide=guide, 
                           activation="softmax")
model = model.to(device)
# Loss an optimizer
# loss = LsLoss()
penaltys = [0,penalty]
loss = GuidedCELoss(penaltys=penaltys,guide=guide)
loss_nop = GuidedCELoss(penaltys=penaltys, guide=False)
optimizer = AdamW(params=model.parameters(), lr=None)
ploss_history = np.zeros(total_iters)
loss_history = np.zeros(total_iters)
compare_history = np.zeros(total_iters)

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
# total_iters= 2
while iter_num < total_iters: 
    optimizer.zero_grad() 
    _, res = sampler.get_batch(device=device, batch_size=batch_size, guide=guide)
    out = model(res[0])
    output = loss(out, [res[1], res[2]])
    output.backward() 
    output_nop = loss_nop(out, [res[1], res[2]])
    ploss_history[iter_num] = output.item()

    loss_history[iter_num] = output_nop.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
    lr = get_lr_cosine_schedule(iter_num, lr_max, lr_min, warmup_iters, total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    if iter_num>0 and iter_num % log_interval == 0:
        # logger.info(f'Iter: {iter_num},Penalty train loss: {np.mean(ploss_history[iter_num//2:iter_num]):.4f}, Train loss: {np.mean(loss_history[iter_num//2:iter_num]):.4f}, Penalty: [{output[1]:.4f}, {output[2]:.4f},{output[3]:.4f},{output[4]:.4f}], Compare: {np.mean(compare_history[iter_num//2: iter_num]):.4f},  Bayes:{Bayes_loss:.4f}, LR: {lr:.6f}, Time: {(finish_time - curr_time):.2f}s')
        logger.info(f'Iter: {iter_num},Penalty train loss: {np.mean(ploss_history[iter_num//2:iter_num]):.4f}, Train loss: {np.mean(loss_history[iter_num//2:iter_num]):.4f}, Penalty: [{output.item():.4f}],  Bayes:{Bayes_loss:.4f}, LR: {lr:.6f}, Time: {(finish_time - curr_time):.2f}s')
        if not raw:
            wandb.log({'train_loss': loss_history[iter_num],
                        'penalty_train_loss':ploss_history[iter_num], 
                        # 'Compare':compare_history[iter_num], 
                        'lr': lr, 'Bayes_loss': Bayes_loss, 
                        'Bayes_std': Bayes_std, 'iter': iter_num})
                        
    
    if iter_num % eval_interval == 0 and not raw:
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                'loss': loss, 'iter': iter_num}, checkpoint_path)
    iter_num += 1


