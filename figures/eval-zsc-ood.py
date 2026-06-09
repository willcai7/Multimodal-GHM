"""Generate zero-shot-classification out-of-distribution risk data for Fig. 8b."""

import torch
import numpy as np
from ghmclip.models.model import EncoderTransformer, GuidedClipLoss
from ghmclip.data.data_random_GHM import DoubleSampler, ClipSampler, PPCLIPLoss
import pathlib
from collections import defaultdict
import json
from eval_paths import checkpoint_dir, ghm_output_path, latest_checkpoint

BATCH_SIZE = 250
MODEL_FOLDER_DICT = {
    "Standard TF": "TF_L5H4D128_L5H4D128",
    "Guided TF": "GT_L5H4D128_L5H4D128",
    "Shallow TF": "TF_L1H4D128_L1H4D128"
}
FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
CLIP_FOLDER = checkpoint_dir("CLIP")
DEVICE="cpu"


def root_to_first_text_leaf_pp(root_pp, text_transition):
    """Project root posterior to the first text leaf using a text transition."""
    leaf_pp = root_pp
    for layer_pps in text_transition:
        leaf_pp = leaf_pp @ layer_pps[0]
    return leaf_pp


def load_model(path_run, device="cuda"):
    """Load a trained CLIP text/image encoder pair for zero-shot evaluation."""
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    n_model_layer = 1 if model_name == "Shallow TF" else 5

    text_model = EncoderTransformer(n_token=81,
                                    num_class=10,
                                    n_layer=n_model_layer,
                                    n_guided_layer=n_model_layer
                                    )

    image_model = EncoderTransformer(n_token=81,
                                    num_class=10,
                                    n_layer=n_model_layer,
                                    n_guided_layer=n_model_layer
                                    )
    
    text_model.load_state_dict(ckpt["tmodel_state_dict"])
    image_model.load_state_dict(ckpt["imodel_state_dict"])
    text_model = text_model.to(device)
    image_model = image_model.to(device)
    return text_model, image_model


def main():
    """Evaluate zero-shot CLIP classification on mismatched p-flip samplers."""
    # Initialize the mis-specified belief-propagation tree.
    # CLIP checkpoints are fixed at p=0.20; only the evaluation sampler varies.
    p_ys = [np.ones(10)/10, np.ones(10)/10]
    p_flips = [0.2, 0.2]
    n_layers = [4, 4]
    n_childs = [3, 3]
    tree_sampler = DoubleSampler(n_layers, n_childs, p_ys, p_flips)
    text_tree, image_tree = tree_sampler.get_zeroshot_batch(batch_size=BATCH_SIZE*30, return_tree=True)
    
    # Init models 
    model_set = "K4_L4C3p20_L4C3p20sc10"
    path_runs = CLIP_FOLDER / model_set
    # Load CLIP variants once and reuse them across all OOD p values.
    model_dicts = {model_name: load_model(path_runs / path_run, device=DEVICE) for model_name, path_run in MODEL_FOLDER_DICT.items()}
    loss = GuidedClipLoss(4, BATCH_SIZE, penalty=0, guide=False)

    # Loop over test p values.
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist() if isinstance(p_list, np.ndarray) else p_list
    for p in p_list:

        # Build a zero-shot batch from the current OOD test distribution.
        sampler = DoubleSampler(n_layers, n_childs, p_ys, [p/100, p/100])
        total_samples = BATCH_SIZE * 30
        t_leaves, i_leaves, t_pp, i_pp, root = sampler.get_zeroshot_batch(batch_size=total_samples)
        i_leaves = torch.tensor(i_leaves, dtype=torch.long).to(DEVICE)
        t_leaves = torch.tensor(t_leaves, dtype=torch.long).to(DEVICE)
        i_pp = root_to_first_text_leaf_pp(i_pp, sampler.t_transition)
        i_pp = torch.tensor(i_pp, dtype=torch.float).to(DEVICE)
        i_pp = torch.log(i_pp)
        true_label = t_leaves[:,0]
        bayes_loss = torch.nn.functional.cross_entropy(i_pp, true_label)
        print("#"*40)
        print(f"p: {p}, Bayes Loss: {bayes_loss.item()}")
        res_dict["Bayes"].append(bayes_loss.item())


        image_tree.T_value[-1] = [i_leaves[:,idx].tolist() for idx in range(81)]
        image_tree.build_tree() 
        image_tree.BP_CLS()
        i_pp = image_tree.posterior_probability_CLS.T
        i_pp = root_to_first_text_leaf_pp(i_pp, sampler.transition)
        i_pp = torch.tensor(i_pp, dtype=torch.float).to(DEVICE)
        i_pp = torch.log(i_pp)
        misspec_loss = torch.nn.functional.cross_entropy(i_pp, true_label)
        res_dict["Mis-spec. BP"].append(misspec_loss.item())
        print(f"p: {p}, Mis.spe. BP Loss: {misspec_loss.item()}")

        text_samples_by_first_coor = {}

        for i in range(10):
            index = torch.where(t_leaves[:,0]==i)[0]
            text_samples_by_first_coor[i] = index
            assert index.numel() >= BATCH_SIZE, f"Class {i} only has {index.numel()} text samples"
        # Compute zero-shot losses using class prototypes from sampled text leaves.
        for model_name, (t_model, i_model) in model_dicts.items():
            t_model.eval()
            i_model.eval()
            t_output,_ = t_model(t_leaves)
            i_output,_ = i_model(i_leaves)

            exp_similarity = torch.exp(i_output @ t_output.T)
            model_predict = torch.zeros(total_samples,10).to(DEVICE)
            for i in range(10):
                index = text_samples_by_first_coor[i]
                index = index[:250]
                sub_similarity = torch.log(exp_similarity[:, index].mean(dim=1))
                model_predict[:,i] = sub_similarity 
            loss = torch.nn.functional.cross_entropy(model_predict, true_label)
            res_dict[model_name].append(loss.item())
            print(f"p: {p}, {model_name} Loss: {loss.item()}")
        
    # The OOD plotting notebook reads this canonical JSON filename.
    with open(ghm_output_path("zsc-ood.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
