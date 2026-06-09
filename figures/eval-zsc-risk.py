"""Generate zero-shot-classification in-distribution risk data for Fig. 2b."""

import torch
import pathlib
from ghmclip.models.model import EncoderTransformer
from ghmclip.data.data_random_GHM import DoubleSampler
from collections import defaultdict
import numpy as np
import json
from eval_paths import checkpoint_dir, ghm_output_path, latest_checkpoint

model_folder_dict = {
    "Standard TF": "TF_L5H4D128_L5H4D128",
    "Guided TF": "GT_L5H4D128_L5H4D128",
    "Shallow TF": "TF_L1H4D128_L1H4D128"
}

folder_model_dict = {v: k for k, v in model_folder_dict.items()}


def load_model(path_run, device="cuda"):
    """Load a trained CLIP text/image encoder pair for a model variant."""
    folder_name = path_run.name
    assert folder_name in folder_model_dict, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = folder_model_dict[folder_name]
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    n_model_layer = 1 if model_name == "Shallow TF" else 5

    text_model = EncoderTransformer(n_token=81,
                                    num_class=10,
                                    n_embd=128,
                                    n_layer=n_model_layer,
                                    n_guided_layer=n_model_layer,
                                    n_head=4,
                                    n_mlp_multiplier=4,
                                    activation="softmax",
                                    mlp=True,
                                    normalize_attn=True,
                                    layernorm=True,
                                    guide=False)

    image_model = EncoderTransformer(n_token=81,
                                    num_class=10,
                                    n_embd=128,
                                    n_layer=n_model_layer,
                                    n_guided_layer=n_model_layer,
                                    n_head=4,
                                    n_mlp_multiplier=4,
                                    activation="softmax",
                                    mlp=True,
                                    normalize_attn=True,
                                    layernorm=True,
                                    guide=False)
    
    text_model.load_state_dict(ckpt["tmodel_state_dict"])
    image_model.load_state_dict(ckpt["imodel_state_dict"])
    text_model = text_model.to(device)
    image_model = image_model.to(device)
    return text_model, image_model

def zsc_loss(sampler, model_dicts, num_samples_list, device="cuda"):
    """Estimate zero-shot cross entropy for each requested support size."""
    # Draw enough shared-root text/image samples to provide every requested
    # support size for every class.
    total_samples = max(num_samples_list) * 30
    t_leaves, i_leaves, t_pp, i_pp, root = sampler.get_zeroshot_batch(batch_size=total_samples)
    t_transtion = sampler.t_transition
    i_leaves = torch.tensor(i_leaves, dtype=torch.long).to(device)
    t_leaves = torch.tensor(t_leaves, dtype=torch.long).to(device)
    for layer_pps in t_transtion:
        # Project the image root posterior through the text transition so the
        # Bayes baseline predicts the first text leaf label.
        i_pp = i_pp @ layer_pps[0]
    i_pp = torch.tensor(i_pp, dtype=torch.float).to(device)
    i_pp = torch.log(i_pp)
    res = defaultdict(list)
    res["num_samples_list"] = num_samples_list.tolist()

    true_label = t_leaves[:,0]
    bayes_loss = torch.nn.functional.cross_entropy(i_pp, true_label)
    print(f" Bayes Loss: {bayes_loss.item()}")
    res["Bayes"].append(bayes_loss.item())

    text_samples_by_first_coor = {}

    for i in range(10):
        # Build class prototypes from text samples whose first coordinate is i.
        index = torch.where(t_leaves[:,0]==i)[0]
        text_samples_by_first_coor[i] = index
        assert index.numel() >= max(num_samples_list), f"Class {i} only has {index.numel()} text samples"

    for model_name, (t_model, i_model) in model_dicts.items():
        t_model.eval()
        i_model.eval()
        
        i_embeddings = torch.zeros(total_samples, 10).to(device)
        t_embeddings = torch.zeros(total_samples, 10).to(device)
        minibatch_size =200
        accumulated_steps = total_samples // minibatch_size
        for step in range(accumulated_steps):
            start_idx = step * minibatch_size
            end_idx = start_idx + minibatch_size
            i_embeddings[start_idx:end_idx],_ = i_model(i_leaves[start_idx:end_idx])
            t_embeddings[start_idx:end_idx],_ = t_model(t_leaves[start_idx:end_idx])

        exp_similarity = torch.exp(i_embeddings @ t_embeddings.T)

        for num_samples in num_samples_list:
            model_predict = torch.zeros(total_samples,10).to(device)
            for i in range(10):
                # Average similarities to the first num_samples text prototypes
                # of class i, then use these scores as zero-shot logits.
                index = text_samples_by_first_coor[i][:num_samples]
                sub_similarity = torch.log(exp_similarity[:, index].mean(dim=1))
                model_predict[:,i] = sub_similarity 
            
            true_label = t_leaves[:,0]
            loss = torch.nn.functional.cross_entropy(model_predict, true_label)
            print(f"{model_name} {num_samples} loss: {loss.item()}")
            res[model_name].append(loss.item())
    
    return res

if __name__ == "__main__":
    # Sweep p-flip with a fixed zero-shot support size.
    device = "cpu"
    clip_folder = checkpoint_dir("CLIP")
    p_flip_list = np.arange(2, 42, 2)
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_flip_list.tolist()
    for p_flip in p_flip_list:
        # Each p setting uses the matching CLIP checkpoint folder and sampler.
        sampler = DoubleSampler(n_layers=[4,4], 
                        n_childs=[3,3], 
                        variable_type=10,
                        p_ys=[np.ones(10)/10, np.ones(10)/10], 
                        p_flips=[p_flip/100, p_flip/100],
                        seedtree=42)
        data_name = f"K4_L4C3p{p_flip}_L4C3p{p_flip}sc10"
        path_runs = clip_folder / data_name
        model_dicts = {model_name: load_model(path_runs / path_run, device=device) for model_name, path_run in model_folder_dict.items()}
        num_samples_list = np.array([250])
        zsc_res =zsc_loss(sampler, model_dicts, num_samples_list, device=device)
        for model_name in model_folder_dict.keys():
            res_dict[model_name].append(zsc_res[model_name][0])
        res_dict["Bayes"].append(zsc_res["Bayes"][0])
    # The task-risk plotting notebook reads this canonical JSON filename.
    with open(ghm_output_path("zsc-risk.json"), "w") as f: 
        json.dump(res_dict, f, indent=4)
