"""
Tests of functions, classes, and models.
"""
import os 
import sys
import numpy as np
import pytest
from ghmclip.data.data_random_GHM import *
from ghmclip.models.models import *
from ghmclip.models.functional import *

@pytest.fixture
def common_params():
    return {
        'n_layers': (3,3),
        'n_childs': (2,2),
        'p_ys': (torch.ones(5)/5, torch.ones(5)/5),
        'p_flips': (0.2, 0.2),
        'flip_scale': 1,
        'variable_type': 5,
        'sigma': 1,
        'translation_invariance': True,
        'batch_size': 10000,
        'seedtree': [42, 24],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def test_double_sampler(common_params):
    params = common_params 
    sampler = DoubleSampler(n_layers=params['n_layers'], 
                            n_childs=params['n_childs'], 
                            p_ys=params['p_ys'], 
                            p_flips=params['p_flips'], 
                            flip_scale=params['flip_scale'], 
                            variable_type=params['variable_type'], 
                            translation_invariance=params['translation_invariance'], 
                            seedtrees=params['seedtree'],
                            device=params['device'])
    batch_text, batch_image = sampler.get_batch(params['batch_size'], job='clip-clip', guide=True)

    # Check the device 
    text_leaves = batch_text[0]
    text_root = batch_text[1]
    text_guided_info = batch_text[2]
    text_pp = batch_text[3]
    image_leaves = batch_image[0]
    image_root = batch_image[1]
    image_guided_info = batch_image[2]
    image_pp = batch_image[3]
    assert text_leaves.device.type == params['device']
    assert image_leaves.device.type == params['device']
    assert text_root.device.type == params['device']
    assert image_root.device.type == params['device']
    assert text_pp.device.type == params['device']
    assert image_pp.device.type == params['device']

    # Check the shape 
    n_leaves = params['n_childs'][0]**params['n_layers'][0]
    assert text_leaves.shape == (params['batch_size'], n_leaves)
    assert image_leaves.shape == (params['batch_size'], n_leaves)
    assert text_root.shape == (params['batch_size'],)
    assert image_root.shape == (params['batch_size'],)
    assert text_pp.shape == (params['batch_size'], params['variable_type'])
    assert image_pp.shape == (params['batch_size'], params['variable_type'])

    # Check the prosterior probability 
    ## Check mean 
    possible_values = torch.arange(params['variable_type']).to(params['device']).view(-1, 1).float()
    text_root = text_root.float()
    image_root = image_root.float()
    print("text_pp dtype: ", text_pp.dtype)
    print("possible_values dtype: ", possible_values.dtype)
    text_mean = text_pp @ possible_values
    image_mean = image_pp @ possible_values
    text_mean = text_mean.flatten()
    image_mean = image_mean.flatten()
    assert torch.linalg.norm(text_mean.mean() - text_root.mean()) < 3e-2
    assert torch.linalg.norm(image_mean.mean() - image_root.mean()) < 3e-2

    ## Check power 
    text_power = text_mean * text_root 
    image_power = image_mean * image_root 
    text_power_pos = text_mean * text_mean 
    image_power_pos = image_mean * image_mean 
    print(text_power[:10])
    print(text_power_pos[:10])
    assert torch.linalg.norm(text_power.mean() - text_power_pos.mean()) < 3e-2
    assert torch.linalg.norm(image_power.mean() - image_power_pos.mean()) < 3e-2

    # Check the dimension of clip model 
    guide_dim = gen_guide_indices(job='clip', n_tree_layer=params['n_layers'][0], n_class=params['variable_type'])
    print("guide_dim: ", guide_dim)
    text_model = EncoderTransformer(
        n_head=1,
        d_model=128,
        d_ff=512,
        n_layer=5,
        guide=True,
        n_class=params['variable_type'],
        n_token = n_leaves,
        auto_regressive=False,
        job='cls',
        guide_dim=guide_dim,
    )

    text_model.to(params['device'])
    text_output,text_guided_output, feature = text_model(text_leaves) 

    assert text_output.shape == (params['batch_size'], params['variable_type'])
    for i in range(len(text_guided_output)):
        assert text_guided_output[i].shape == text_guided_info[i].shape

    # Check another kind of guide setting
    batch_text, batch_image = sampler.get_batch(params['batch_size'], job='dns-nwp', guide=True)
    # Check the device 
    input_text = batch_text[0]
    target_text = batch_text[1]
    text_guided_info = batch_text[2]
    text_pp = batch_text[3]
    input_image = batch_image[0]
    target_image = batch_image[1]
    image_guided_info = batch_image[2]
    image_pp = batch_image[3]
    assert input_text.device.type == params['device']
    assert target_text.device.type == params['device']
    assert text_pp.device.type == params['device']
    assert input_image.device.type == params['device']
    assert target_image.device.type == params['device']
    assert image_pp.device.type == params['device']

    # Check the shape 
    n_leaves = params['n_childs'][0]**params['n_layers'][0]
    assert input_text.shape == (params['batch_size'], n_leaves-1)
    assert target_text.shape == (params['batch_size'], n_leaves-1)
    assert input_image.shape == (params['batch_size'], n_leaves)
    assert target_image.shape == (params['batch_size'], n_leaves)
    assert text_pp.shape == (params['batch_size'], n_leaves-1,params['variable_type'])
    assert image_pp.shape == (params['batch_size'], n_leaves)

    # Check the guided info dimension
    for i in range(len(text_guided_info)):
        info = text_guided_info[i] 
        print("info shape: ", info.shape) 
    
    for i in range(len(image_guided_info)):
        info = image_guided_info[i] 
        print("info shape: ", info.shape) 

    for i in range(len(text_guided_info)):
        if i< len(text_guided_info)//2-1:
            assert text_guided_info[i].shape == (params['batch_size'], n_leaves-1, 2*params['variable_type'])
        else:
            assert text_guided_info[i].shape == (params['batch_size'], n_leaves-1, params['variable_type'])

    for i in range(len(image_guided_info)-1):
        assert image_guided_info[i].shape == (params['batch_size'], n_leaves, params['variable_type'])
    assert image_guided_info[-1].shape == (params['batch_size'], n_leaves)

    

    

