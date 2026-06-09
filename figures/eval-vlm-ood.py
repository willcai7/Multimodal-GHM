"""Generate next-word-prediction OOD risk data for Figs. 8d and 9b."""

import torch
import numpy as np
from ghmclip.models.model import AutoRegressiveTransformer, ConditionalGuidedCELoss, EncoderTransformer
from ghmclip.data.data_random_GHM import DoubleSampler, NextWordPredictSampler
from pathlib import Path
from collections import defaultdict
import json
import torch.nn as nn
from eval_paths import checkpoint_dir, ghm_output_path, latest_checkpoint

BATCH_SIZE = 1000
MODEL_FOLDER_DICT = {
    "Standard TF": "StT_L9H4D256",
    "Guided TF": "GT_L9H4D256",
    "Shallow TF": "ShT_L1H4D256",
    "Joint Training": "JT_L9H4D256",
}

FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
VLM_FOLDER = checkpoint_dir("VLM")
CLIP_FOLDER = checkpoint_dir("CLIP")
DEVICE="cpu"

def CEloss(inputs, targets):
        """Return token-level cross entropy for flattened sequence predictions."""
        clas_inputs = inputs.reshape(-1, inputs.size(-1)) # classification input
        clas_targets = targets.reshape(-1)
        loss = nn.functional.cross_entropy(clas_inputs, clas_targets, reduction='none') # cross entropy loss
        loss = loss.reshape(-1, targets.shape[1])
        loss = torch.mean(loss, dim=1)
        return loss.mean()

def load_clip_image_model(path_run, device="cuda"):
    """Load the CLIP image encoder used by sequential VLM checkpoints."""
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    model = EncoderTransformer(n_token=81,
                               num_class=10,
                               n_layer=5,
                               n_guided_layer=1)
    model.load_state_dict(ckpt["imodel_state_dict"])
    model = model.to(device)
    return model

def load_model(path_run, device="cuda"):
    """Load one trained sequential next-word-prediction model."""
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    print("Loading model: ", model_name)
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    n_model_layer = 1 if model_name == "Shallow TF" else 9
    n_token = 161 if model_name == "Joint Training" or model_name == "Guided TF" else 81
    sequential = False if model_name == "Joint Training" or model_name == "Guided TF" else True
    model = AutoRegressiveTransformer(n_token=n_token,
                                    n_i_token=n_token-80,
                                    num_class=10,
                                    n_embd=256,
                                    n_layer=n_model_layer,
                                    n_guided_layers=[1, 1],
                                    auto_regressive=True,
                                    n_head=4,
                                    sequential=sequential,
                                    n_mlp_hidden=4*256)

    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    return model

def main():
    """Evaluate VLM checkpoints for both-modality and fixed-image OOD sweeps."""
    # Shared tree/task parameters for the trained VLM checkpoint family.
    p_ys = [np.ones(10)/10, np.ones(10)/10]
    p_flips = [0.2, 0.2]
    n_layers = [4, 4]
    n_childs = [3, 3]
    
    # Initialize trained VLM models.
    model_set = "K4_L4C3p20_L4C3p20sc10"
    path_runs = VLM_FOLDER / model_set
    # Load all VLM variants once because only the test sampler changes.
    model_dicts = {model_name: load_model(path_runs / path_run, device=DEVICE) for model_name, path_run in MODEL_FOLDER_DICT.items()}

    path_clip = CLIP_FOLDER / "K4_L4C3p20_L4C3p20sc10" / "TF_L5H4D128_L5H4D128"
    # Sequential VLM variants use this CLIP image encoder as a frozen feature
    # extractor; joint variants consume raw image leaves.
    clip_image_model = load_clip_image_model(path_clip, device=DEVICE)
    criterion = CEloss

    def evaluate_sweep(sweep_name, output_name, p_pair_for_plot_value):
        """Run one VLM OOD sweep and write the JSON consumed by a notebook."""
        # The belief-propagation baseline remains mis-specified at the training
        # p=0.20, while the sampled evaluation data use the p pair below.
        tree_sampler = DoubleSampler(n_layers, n_childs, p_ys, p_flips)
        text_tree, image_tree = tree_sampler.get_zeroshot_batch(batch_size=BATCH_SIZE, return_tree=True)

        p_list = np.arange(2, 42, 2)
        res_dict = defaultdict(list)
        res_dict["p_flip"] = p_list.tolist() if isinstance(p_list, np.ndarray) else p_list
        for p in p_list:
            pt, pi = p_pair_for_plot_value(p)
            # Build an OOD next-word-prediction batch for the current p pair.
            sampler = NextWordPredictSampler(n_layers, n_childs, p_ys, [pt/100, pi/100])

            Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
            print("#"*40)
            print(f"{sweep_name}: plot p={p}, p_t={pt}, p_i={pi}")
            print(f"Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}")
            res_dict["Bayes"].append(Bayes_loss.item())
            res_text, res_image= sampler.get_batch(device=DEVICE, batch_size=BATCH_SIZE, guide=False)
            # Replace the tree leaves with the current sampled batch before BP.
            for idx in range(80):
                text_tree.T_value[-1][idx] = res_text[0][:,idx].tolist()
            image_tree.T_value[-1] = [res_image[0][:,idx].tolist() for idx in range(81)]
            text_tree.build_tree()
            image_tree.build_tree()
            image_tree.BP_CLS()
            external_hd_message = image_tree.root_node.hd_message
            BP_output,_ = text_tree.BP_NWP_autoregressive(external_hd_message=external_hd_message, device=DEVICE, guide_info=False)
            pred = BP_output
            print(f"BP output shape: {pred.shape}")
            target = res_text[1]
            pred = pred.reshape(-1, 10)
            target_c = target.reshape(-1)
            loss = -torch.log(pred[range(len(target_c)), target_c])
            loss = torch.mean(loss)
            print(f"Mis.spe. BP Loss: {loss}")
            res_dict["Mis-spec. BP"].append(loss.item())

            # Compute model losses using either raw image leaves or CLIP image features.
            for model_name, model in model_dicts.items():
                if model_name == "Shallow TF" or model_name == "Standard TF":
                    image_input = clip_image_model(res_image[0])[0].unsqueeze(1)
                else:
                    image_input = res_image[0]
                image_output = model(res_text[0], image_input)
                loss_out = criterion(image_output[0], res_text[1])
                print(f"{model_name} Loss: {loss_out.item()}")
                res_dict[model_name].append(loss_out.item())

        with open(ghm_output_path(output_name), "w") as f:
            json.dump(res_dict, f, indent=4)

    # Fig. 8d changes both modalities together away from the training p=0.20.
    evaluate_sweep("Fig. 8d VLM both-modality OOD", "vlm-ood.json", lambda p: (p, p))
    # Fig. 9b keeps image fixed at p_i=0.20 and changes only the text p_t.
    evaluate_sweep("Fig. 9b VLM fixed-image OOD", "vlm-ood-pi20.json", lambda p: (p, 20))

if __name__ == "__main__":
    main()
