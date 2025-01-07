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

from src.data import NextWordPredictSampler
from src.models import *
from src.utils import * 

# Generate the Config class for the training script
@dataclass 
class TrainingConfig(UtilConfig, DoubleTreeConfig, ModelConfig):
    clip_feature: Optional[str] = field(default='GT')
    job_name: Optional[str] = field(default='Sequential_NWP')

parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
config_dict = vars(config)
locals().update(config_dict)

# CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Model Input Dimension 
d_tmodel = n_ttree_child**n_ttree_layer
d_imodel = n_itree_child**n_itree_layer
d_model =d_tmodel
# Model 
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
tree_folder = f'K{K}_L{n_ttree_layer}C{n_ttree_child}p{int(p_ttree_flip*100)}_L{n_itree_layer}C{n_itree_child}p{int(p_itree_flip*100)}sc{int(flip_scale*10)}'
model_name = f'L{n_model_layer}H{n_head}D{d_eb}'
tags=[job_name, tree_folder]
if guide: 
    tags.append('guide')
    model_name = 'GT_' + model_name
else: 
    model_name = 'TF_' + model_name

if clip_feature == 'TF':
    model_name = 'V' + model_name

# Initialize Loggers 
directory = os.path.join("./logs", job_name, tree_folder, model_name, timestamp) 
logger = GenLogger(directory, config, raw=raw)
if not raw:
    wandb.init(project=wandb_project, name = timestamp + '-' + model_name, tags=tags, dir=wandb_path)
    wandb.config.update(asdict(config)) 
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

# sampler 

p_y = np.ones(variable_type) / variable_type 
sampler = NextWordPredictSampler([n_ttree_layer, n_itree_layer], 
                                [n_ttree_child, n_itree_child], 
                                [p_y, p_y], 
                                [p_ttree_flip, p_itree_flip], 
                                flip_scale=flip_scale,
                                variable_type=variable_type, 
                                translation_invariance=True, 
                                seedtree=42)


if not raw:
    Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
    logger.info(f'Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}')    
    wandb.log({'Bayes_loss': Bayes_loss, 'Bayes_std': Bayes_std})
else:
    Bayes_loss = 0
    Bayes_std = 0

# Clip image model
clip_image_model = EncoderTransformer(n_token=d_imodel,
                            num_class=variable_type,
                            n_embd=128,
                            n_layer=5,
                            n_head=4,
                            n_mlp_multiplier=4,
                            activation=activation,
                            mlp=True,
                            normalize_attn=True,
                            layernorm=True,
                            maxnorm=False,
                            guide=False)

clip_path = os.path.join("saved_models", "clip","softmax_attention", tree_folder)
model_folders = os.listdir(clip_path)
for folder in model_folders:
    if "GT" in folder and clip_feature == "GT":
    # if "TF" in folder and "L5" in folder:
        clip_path = os.path.join(clip_path, folder)
        break
    elif "TF" in folder and "L5" in folder and clip_feature == "TF":
        clip_path = os.path.join(clip_path, folder)
        break
clip_path = os.path.join(clip_path, os.listdir(clip_path)[0])
clip_checkpoint_path = os.path.join(clip_path, "checkpoint.pth")
checkpoint = torch.load(clip_checkpoint_path, map_location=device) 
clip_image_model.load_state_dict(checkpoint['imodel_state_dict']) # load the state dict for text model
clip_image_model = clip_image_model.to(device)

# Model 
seed_everything(seed)
model = AutoRegressiveTransformer(n_token=d_model,
                                n_i_token=1,
                                num_class=variable_type,
                                n_embd=d_eb,
                                n_layer=n_model_layer,
                                n_guided_layers=[n_ttree_layer,1],
                                n_head=n_head,
                                n_mlp_hidden=4*d_eb,
                                auto_regressive=True,
                                activation="softmax",
                                mlp=True,
                                normalize_attn=normalize_attn,
                                layernorm=layernorm,
                                sequential=True,
                                guide=guide)
model = model.to(device)
loss = ConditionalGuidedCELoss(penalty=penalty,guide=guide)
loss_nop = ConditionalGuidedCELoss(penalty=0,guide=False)
compare_measure = KLdiv()
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
    res_text, res_image = sampler.get_batch(device=device, batch_size=batch_size, guide=guide)
    clip_image_output = clip_image_model(res_image[0])[0].unsqueeze(1)
    guided_layers = [res_text[-2], [clip_image_output,clip_image_output]]
    posterior = res_text[-1].to(device)
    
    out = model(res_text[0], clip_image_output)

    
    output = loss(out, [res_text[1], guided_layers], verbose=False)
    output[0].backward() 
    output_nop = loss_nop(out, [res_text[1], guided_layers])
    # print(output_nop.item())
    output_compare = compare_measure(out[0], posterior)

    ploss_history[iter_num] = output[0].item()
    loss_history[iter_num] = output_nop[0].item()
    compare_history[iter_num] = output_compare.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
    lr = get_lr_cosine_schedule(iter_num, lr_max, lr_min, warmup_iters, total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    if iter_num>0 and iter_num % log_interval == 0:
        logger.info(
            f"Iter: {iter_num}, "
            f"Penalty train loss: {np.mean(ploss_history[iter_num // 2 : iter_num]):.4f}, "
            f"Train loss: {np.mean(loss_history[iter_num // 2 : iter_num]):.4f}, "
            f"Penalty: [{output[1]:.4f}, {output[2]:.4f}, {output[3]:.4f}, {output[4]:.4f}], "
            f"Compare: {np.mean(compare_history[iter_num//2: iter_num]):.4f},"
            f"Bayes: {Bayes_loss:.4f}, "
            f"LR: {lr:.6f}, "
            f"Time: {(finish_time - curr_time):.2f}s",
        )        
        if not raw:
            wandb.log({'train_loss': loss_history[iter_num],
                        'penalty_train_loss':ploss_history[iter_num], 
                        'Compare':compare_history[iter_num], 
                        'lr': lr, 'Bayes_loss': Bayes_loss, 
                        'Bayes_std': Bayes_std, 'iter': iter_num})
                        
    
    if iter_num % eval_interval == 0 and not raw:
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                'loss': loss, 'iter': iter_num, 'loss_history':loss_history, 'ploss_history':ploss_history, 'bayes':Bayes_loss, 'compare':compare_history}, checkpoint_path)
    iter_num += 1

logging.shutdown() # close the logger

if not raw:
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                'loss': loss, 'iter': iter_num, 'loss_history':loss_history, 'ploss_history':ploss_history, 'bayes':Bayes_loss, 'compare':compare_history}, checkpoint_path)

if S3_upload: 
    import s3fs
    s3_file = s3fs.S3FileSystem()
    local_path = directory
    s3_path = S3_bucket_name+f'/GHM/{job_name}/{tree_folder}/{model_name}/{timestamp}'
    s3_file.put(local_path, s3_path, recursive=True) 