import torch
import numpy as np
from ghmclip.models.model import EncoderTransformer, GuidedClipLoss
from ghmclip.data.data_random_GHM import DoubleSampler, ClipSampler, PPCLIPLoss
import pathlib
from collections import defaultdict
import json

BATCH_SIZE = 1000
MODEL_FOLDER_DICT = {
    "Standard TF": "TF_L5H4D128_L5H4D128",
    "Guided TF": "GT_L5H4D128_L5H4D128",
    "Shallow TF": "TF_L1H4D128_L1H4D128"
}
FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
CLIP_FOLDER = pathlib.Path("/data01/home/yuhang.cai/Multimodal-GHM/logs/CLIP/")
DEVICE="cpu"

def load_history(path_run, device="cuda"):
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    path_run_timestamp = path_run.glob("*")
    path_run_timestamp = list(path_run_timestamp)[0]
    path_ckpt = path_run_timestamp / "checkpoint.pth"
    if not path_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path_ckpt}")
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    return ckpt["loss_history"][-100:].mean(), ckpt["bayes"]

def main():
    # Loop over p 
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist()
    for p in p_list:
        model_set = f"K4_L4C3p{p}_L4C3p{p}sc10"
        path_runs = CLIP_FOLDER / model_set
        for model_name, path_run in MODEL_FOLDER_DICT.items():
            loss_history, bayes = load_history(path_runs / path_run, device=DEVICE)
            print(f"p: {p}, {model_name} Loss: {loss_history}, Bayes: {bayes}")
            res_dict[model_name].append(loss_history)
        res_dict["Bayes"].append(bayes)

        
    with open("./figures/data/clip-risk.json", "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
