import os
import sys

src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(src_path)

import torch
import numpy as np
from src.models.models import EncoderTransformer
import pytest

@pytest.fixture
def test_params():
    return {
        'batch_size': 128,
        'n_token': 8,
        'n_class': 10,
        'n_embd': 128,
        'n_layer': 6,
        'n_head': 4,
        'guided_dim': [3, 5, 2]
    }

def test_encoder_transformer_dimensions(test_params):
    """Test output dimensions of EncoderTransformer"""
    params = test_params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    model = EncoderTransformer(
        n_token=params['n_token'],
        n_head=params['n_head'],
        d_model=params['n_embd'],
        d_ff=params['n_embd'] * 4,
        n_layer=params['n_layer'],
        guide=True,
        n_class=params['n_class'],
        guide_dim=params['guided_dim']
    )
    model.to(device)
    # Create test input
    x = torch.randint(0, params['n_class'], (params['batch_size'], params['n_token'])).to(device)
    
    # Get model outputs
    prediction, guided_layers, features = model(x, feature=True)
    
    # Test input shape
    assert x.shape == (params['batch_size'], params['n_token'])
    
    # Test prediction shape
    assert prediction.shape == (params['batch_size'], params['n_class'])
    
    # Test number of guided layers
    assert len(guided_layers) == len(params['guided_dim'])
    
    # Test guided layer shapes
    for i, layer in enumerate(guided_layers):
        assert layer.shape == (params['batch_size'], params['n_token'], params['guided_dim'][i])
    
    # Test number of feature layers
    assert len(features) == params['n_layer']
    
    # Test feature layer shapes
    for feat in features:
        assert feat.shape == (params['batch_size'], params['n_token'], params['n_embd'])

