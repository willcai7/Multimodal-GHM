import torch
import numpy as np
from ghmclip.models.model import EncoderTransformer, GuidedClipLoss
from ghmclip.data.data_random_GHM import DoubleSampler, ClipSampler, PPCLIPLoss
import pathlib
from collections import defaultdict
import json

BATCH_SIZE = 5000
MODEL_FOLDER_DICT = {
    "Standard TF": "TF_L5H4D128_L5H4D128",
    "Guided TF": "GT_L5H4D128_L5H4D128",
    "Shallow TF": "TF_L1H4D128_L1H4D128"
}
FOLDER_MODEL_DICT = {v: k for k, v in MODEL_FOLDER_DICT.items()}
CLIP_FOLDER = pathlib.Path("/data01/home/yuhang.cai/Multimodal-GHM/logs/CLIP/")
DEVICE="cpu"

def load_model(path_run, device="cuda"):
    folder_name = path_run.name
    assert folder_name in FOLDER_MODEL_DICT, f"Folder name {folder_name} not found in folder_model_dict"
    model_name = FOLDER_MODEL_DICT[folder_name]
    path_run_timestamp = path_run.glob("*")
    path_run_timestamp = list(path_run_timestamp)[0]
    path_ckpt = path_run_timestamp / "checkpoint.pth"
    if not path_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path_ckpt}")
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
    # Init mis.specified BP tree
    p_ys = [np.ones(10)/10, np.ones(10)/10]
    p_flips = [0.2, 0.2]
    n_layers = [4, 4]
    n_childs = [3, 3]
    tree_sampler = DoubleSampler(n_layers, n_childs, p_ys, p_flips)
    text_tree, image_tree = tree_sampler.get_zeroshot_batch(batch_size=BATCH_SIZE*5, return_tree=True)
    
    # Init models 
    model_set = "K4_L4C3p20_L4C3p20sc10"
    path_runs = CLIP_FOLDER / model_set
    model_dicts = {model_name: load_model(path_runs / path_run, device=DEVICE) for model_name, path_run in MODEL_FOLDER_DICT.items()}
    loss = GuidedClipLoss(4, BATCH_SIZE, penalty=0, guide=False)

    # Loop over p 
    p_list = np.arange(2, 42, 2)
    # p_list = [20]
    res_dict = defaultdict(list)
    res_dict["p_flip"] = p_list.tolist()
    for p in p_list:

        # Init sampler 
        sampler = ClipSampler(n_layers, n_childs, p_ys, [p/100, p/100])


        Bayes_loss, Bayes_std = sampler.get_Bayes(n_eval=10000)
        print("#"*40)
        print(f"p: {p}, Bayes Loss: {Bayes_loss}, Bayes Std: {Bayes_std}")
        res_dict["Bayes"].append(Bayes_loss)
        res_text, res_image= sampler.get_batch(device=DEVICE, batch_size=BATCH_SIZE, guide=False)
        text_tree.T_value[-1] = [res_text[0][:,idx].tolist() for idx in range(81)]
        image_tree.T_value[-1] = [res_image[0][:,idx].tolist() for idx in range(81)]
        text_tree.build_tree()
        image_tree.build_tree() 
        text_tree.BP_CLS()
        image_tree.BP_CLS()
        t_pp = text_tree.posterior_probability_CLS
        i_pp = image_tree.posterior_probability_CLS
        mis_spe_bp_loss,_ = PPCLIPLoss(t_pp, i_pp, BATCH_SIZE, K=4, variable_type=10)
        res_dict["Mis-spec. BP"].append(mis_spe_bp_loss)
        print(f"p: {p}, Mis.spe. BP Loss: {mis_spe_bp_loss}")

        # Compute model losses
        for model_name, (t_model, i_model) in model_dicts.items():
            t_model.eval()
            i_model.eval()
            guided_layers = [res_text[2], res_image[2]]
            t_output = t_model(res_text[0])
            i_output = i_model(res_image[0])
            output = loss(t_output, i_output, guided_layers)
            print(f"p: {p}, {model_name} Loss: {output[0].item()}")
            res_dict[model_name].append(output[0].item())
        
    with open("./figures/data/ood-clip.json", "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
