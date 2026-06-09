"""Generate next-word-prediction in-distribution risk data for Fig. 2d."""

import torch
import numpy as np
import pathlib
from collections import defaultdict
import json
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
DEVICE="cpu"

def load_history(path_run, device="cuda"):
    """Load the final loss-history window and Bayes baseline from one VLM run."""
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    return ckpt["loss_history"][-100:].mean(), ckpt["bayes"]

def main():
    """Evaluate VLM checkpoints over the p-flip grid and write vlm-risk.json."""
    # Loop over p.
    # p is stored as an integer percentage in folder names, e.g. p20 = 0.20.
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist()
    for p in p_list:
        model_set = f"K4_L4C3p{p}_L4C3p{p}sc10"
        path_runs = VLM_FOLDER / model_set
        # Load the final loss history for each VLM architecture variant.
        for model_name, path_run in MODEL_FOLDER_DICT.items():
            loss_history, bayes = load_history(path_runs / path_run, device=DEVICE)
            print(f"p: {p}, {model_name} Loss: {loss_history}, Bayes: {bayes}")
            res_dict[model_name].append(loss_history)
        res_dict["Bayes"].append(bayes.item())

        
    # The plotting notebook reads this canonical JSON filename.
    with open(ghm_output_path("vlm-risk.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
