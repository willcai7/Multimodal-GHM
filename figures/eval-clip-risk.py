"""Generate CLIP in-distribution risk data for Fig. 2a."""

import torch
import numpy as np
from ghmclip.models.model import EncoderTransformer, GuidedClipLoss
from ghmclip.data.data_random_GHM import DoubleSampler, ClipSampler, PPCLIPLoss
import pathlib
from collections import defaultdict
import json
from eval_paths import checkpoint_dir, ghm_output_path, latest_checkpoint

BATCH_SIZE = 1000
MODEL_FOLDER_DICT = {
    "Standard TF": "TF_L5H4D128_L5H4D128",
    "Guided TF": "GT_L5H4D128_L5H4D128",
    "Shallow TF": "TF_L1H4D128_L1H4D128"
}
FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
CLIP_FOLDER = checkpoint_dir("CLIP")
DEVICE="cpu"

def load_history(path_run, device="cuda"):
    """Load the final loss-history window and Bayes baseline from one CLIP run."""
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    path_ckpt = latest_checkpoint(path_run)
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    return ckpt["loss_history"][-100:].mean(), ckpt["bayes"]

def main():
    """Evaluate CLIP checkpoints over the p-flip grid and write clip-risk.json."""
    # Loop over p.
    # p is stored as an integer percentage in folder names, e.g. p20 = 0.20.
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist()
    for p in p_list:
        model_set = f"K4_L4C3p{p}_L4C3p{p}sc10"
        path_runs = CLIP_FOLDER / model_set
        # Each model variant has its own subfolder under the same tree setting.
        for model_name, path_run in MODEL_FOLDER_DICT.items():
            loss_history, bayes = load_history(path_runs / path_run, device=DEVICE)
            print(f"p: {p}, {model_name} Loss: {loss_history}, Bayes: {bayes}")
            res_dict[model_name].append(loss_history)
        res_dict["Bayes"].append(bayes)

        
    # The plotting notebook reads this canonical JSON filename.
    with open(ghm_output_path("clip-risk.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
