import torch
import numpy as np
from ghmclip.models.model import ConditionalDenoiseEncoderTransformer, GuidedClipLoss, EncoderTransformer
from ghmclip.data.data_random_GHM import DoubleSampler, ConditionalDenoiseSampler
from pathlib import Path
from collections import defaultdict
import json
import torch.nn as nn

BATCH_SIZE = 5000
MODEL_FOLDER_DICT = {
    "Standard TF": "StT_L9H4D128",
    "Guided TF": "GT_L9H4D128",
    "Shallow TF": "ShT_L1H4D128",
    "Joint Training": "JT_L9H4D128",
}

FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
CDM_FOLDER = Path("./logs/CDM/")
CLIP_FOLDER = Path("./logs/CLIP/")
DEVICE="cpu"

class LsLoss(nn.Module):
    def __init__(self):
        super(LsLoss, self).__init__()
    
    def forward(self, inputs, targets):
        return torch.sum(torch.pow(inputs - targets, 2), dim=1).mean()

def load_clip_text_model(path_run, device="cuda"):
    path_run_timestamp = path_run.glob("*")
    path_run_timestamp = list(path_run_timestamp)[0]
    path_ckpt = path_run_timestamp / "checkpoint.pth"
    if not path_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path_ckpt}")
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    model = EncoderTransformer(n_token=81,
                               num_class=10,
                               n_layer=5,
                               n_guided_layer=1)
    model.load_state_dict(ckpt["tmodel_state_dict"])
    model = model.to(device)
    return model

def load_model(path_run, device="cuda"):
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    print("Loading model: ", model_name)
    path_run_timestamp = path_run.glob("*")
    path_run_timestamp = list(path_run_timestamp)[0]
    path_ckpt = path_run_timestamp / "checkpoint.pth"
    if not path_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path_ckpt}")
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=False)
    n_model_layer = 1 if model_name == "Shallow TF" else 9
    n_token = 162 if model_name == "Joint Training" or model_name == "Guided TF" else 82
    sequential = False if model_name == "Joint Training" or model_name == "Guided TF" else True
    model = ConditionalDenoiseEncoderTransformer(n_token=n_token,
                                                 n_i_token=81,
                                                 num_class=10,
                                                 n_embd=128,
                                                 n_layer=n_model_layer,
                                                 n_guided_layers=[1, 1],
                                                 n_head=4,
                                                 sequential=sequential,
                                                 n_mlp_hidden=4*128)

    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    return model

def main():
    # Init mis.specified BP tree
    p_ys = [np.ones(10)/10, np.ones(10)/10]
    p_flips = [0.2, 0.2]
    n_layers = [4, 4]
    n_childs = [3, 3]
    tree_sampler = DoubleSampler(n_layers, n_childs, p_ys, p_flips)
    text_tree, image_tree = tree_sampler.get_zeroshot_batch(batch_size=BATCH_SIZE, return_tree=True)
    
    # Init models 
    model_set = "K4_L4C3p20_L4C3p20sc10"
    path_runs = CDM_FOLDER / model_set
    model_dicts = {model_name: load_model(path_runs / path_run, device=DEVICE) for model_name, path_run in MODEL_FOLDER_DICT.items()}

    path_clip = CLIP_FOLDER / "K4_L4C3p20_L4C3p20sc10" / "TF_L5H4D128_L5H4D128"
    clip_text_model = load_clip_text_model(path_clip, device=DEVICE)
    criterion = LsLoss()

    # Loop over p 
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist() if isinstance(p_list, np.ndarray) else p_list
    for p in p_list:
        pt = 20 
        pi = p
        # Init sampler 
        sampler = ConditionalDenoiseSampler(n_layers, n_childs, p_ys, [pt/100, pi/100])

        Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
        print("#"*40)
        print(f"p: {p}, Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}")
        res_dict["Bayes"].append(Bayes_loss)
        res_text, res_image= sampler.get_batch(device=DEVICE, batch_size=BATCH_SIZE, guide=False)
        text_tree.T_value[-1] = [res_text[0][:,idx].tolist() for idx in range(81)]
        image_tree.T_value[-1] = [res_image[1][:,idx].tolist() for idx in range(81)]
        text_tree.build_tree()
        image_tree.build_tree() 
        text_tree.BP_CLS()
        external_hd_message = text_tree.root_node.hd_message
        image_tree.BP_DNS(res_image[0].T.numpy(), 1, external_hd_message=external_hd_message)
        pred = image_tree.posterior_mean_DNS.T
        target = res_image[1].numpy()
        loss = np.sum(np.power(pred-target,2),1)
        loss = np.mean(loss)
        print(f"p: {p}, Mis.spe. BP Loss: {loss}")
        res_dict["Mis-spec. BP"].append(loss.item())
        # Compute model losses
        for model_name, model in model_dicts.items():
            model.eval()
            if model_name == "Shallow TF" or model_name == "Standard TF":
                text_input = clip_text_model(res_text[0])[0].unsqueeze(1)
            else:
                text_input = res_text[0]
            image_output = model(text_input, res_image[0])
            loss_out = criterion(image_output[0], res_image[1])
            print(f"p: {p}, {model_name} Loss: {loss_out.item()}")
            res_dict[model_name].append(loss_out.item())
        
    with open("./figures/data/cdm-ood-pt20.json", "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
