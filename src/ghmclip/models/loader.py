import os
import json
import torch
from ghmclip.models.models import EncoderTransformer




def load_clip_models(path_model, device='cpu', random=False):
    """
    Load the CLIP models from the given path.

    Args:
        path_model: the path to the CLIP models. Should include two files:
            - 'checkpoint.pth': the model parameters
            - 'config.json': the model configuration 
        device: the device to load the models

    Returns:
        clip_model: the CLIP models
    """
    
    # Load configuration
    config_path = os.path.join(path_model, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model parameters from config

    n_t_token = config.get('n_ttree_child') ** config.get('n_ttree_layer')
    n_i_token = config.get('n_itree_child') ** config.get('n_itree_layer')

    text_params = {
        'n_token': n_t_token,
        'n_class': config.get('variable_type', 10),
        'd_model': config.get('clip_tmodel_deb', 128),
        'd_ff': config.get('clip_tmodel_deb', 128)*4,
        'n_layer': config.get('clip_tmodel_nlayer', 12),
        'n_head': config.get('clip_tmodel_nhead', 4),
        'guide': False,
        'job': 'clip',
    }
    
    image_params = {
        'n_token': n_i_token,
        'n_class': config.get('variable_type', 10),
        'd_model': config.get('clip_imodel_deb', 128),
        'd_ff': config.get('clip_imodel_deb', 128)*4,
        'n_layer': config.get('clip_imodel_nlayer', 12),
        'n_head': config.get('clip_imodel_nhead', 4),
        'guide': False,
        'job': 'clip',
    }
    
    # Create text and image models
    text_model = EncoderTransformer(**text_params)
    image_model = EncoderTransformer(**image_params)
    
    if not random:
        # Load model parameters
        checkpoint_path = os.path.join(path_model, 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load state dictionaries, handling potential key mismatches
        text_model.load_state_dict(checkpoint['tmodel_state_dict'])
        image_model.load_state_dict(checkpoint['imodel_state_dict'])
    
    # Move models to device
    text_model = text_model.to(device)
    image_model = image_model.to(device)
    
    # Set models to evaluation mode
    text_model.eval()
    image_model.eval()
    
    return text_model, image_model


def load_model(path_model, device='cpu', random=False):
    """
    Load the single model from the given path.
    """
    config_path = os.path.join(path_model, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)   

    if config.get('job_name') == 'PRH_NWP':
        job = 'nwp'
        n_token = config.get('n_tree_child') ** config.get('n_tree_layer') - 1 
    elif config.get('job_name') == 'PRH_DNS':
        job = 'dns'
        n_token = config.get('n_tree_child') ** config.get('n_tree_layer') 
    else:
        raise ValueError(f"Invalid job name: {config.get('job_name')}")


    model=EncoderTransformer(n_head=1,
                            d_model=config.get('d_eb', 256),
                            d_ff=config.get('d_eb', 256)*4,
                            n_layer=config.get('n_model_layer', 12),
                            n_class=config.get('variable_type', 10),
                            n_token=n_token,
                            auto_regressive=True,
                            job=job)
    
    if not random:
        checkpoint_path = os.path.join(path_model, 'checkpoint.pth')
        # Just load checkpoint directly - sys.modules handling takes care of the remapping
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model




if __name__ == "__main__":
    device = 'cuda:0'
    # path_model = "./saved_models/clip/softmax_attention/K4_L4C3p20_L4C3p20sc10/TF_L5H4D128_L5H4D128/20241121-193933"
    # path_model = "./logs/PRH_NWP/K4_L4C3p60sc10/GT_L9H1D256"
    # path_model = "./logs/PRH_DNS/K4_L4C3p4sc10/GT_L9H1D256"
    path_model = "./logs/PRH_CLIP/K4_L4C3p20_L4C3p20sc10/GT_L5H4D128_L5H4D128"

    path_model = os.path.join(path_model, os.listdir(path_model)[-1])
    text_model, image_model = load_clip_models(path_model, device)
    device = 'cuda:0'
    # input_text = torch.rand.
    # model = load_model(path_model, device)
    # print(model)

    