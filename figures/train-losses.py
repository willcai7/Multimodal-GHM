import torch
import pathlib
import numpy as np
root = pathlib.Path("/data01/home/yuhang.cai/Multimodal-GHM/logs/")
task_types = ["CLIP", "VLM", "CDM"]
p_flip_list = np.arange(2, 42, 2)

model_name_dict = {
    "GT": "Guided TF",
    "JT": "Joint Train",
    "ShT": "Shallow TF",
    "StT": "Standard TF",
    "TF_L1": "Shallow TF",
    "TF_L5": "Standard TF",
}

for task_type in task_types:
    print("Working on task type: ", task_type)
    for p_flip in p_flip_list:
        path_runs = root / task_type / f"K4_L4C3p{p_flip}_L4C3p{p_flip}sc10"
        path_runs = list(path_runs.glob("*"))
        for path_run in path_runs:
            model_set = path_run.name 
            for key, value in model_name_dict.items():
                if key in model_set:
                    model_name = value
                    break
            print(path_run.name, model_name)
        