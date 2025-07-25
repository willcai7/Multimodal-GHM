"""
Tests of functions, classes, and models.
"""
import os 
import sys
src_path = os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(src_path)

import numpy as np
from src.data.data_random_GHM import GenTransition, GHMTree
import pytest

# Common test parameters
@pytest.fixture
def common_params():
    return {
        'n_layer': 3,
        'n_child': 2,
        'p_y': np.ones(10)/10,
        'variable_type': 10,
        'p_flip': 0.1,
        'flip_scale': 1,
        'sigma': 0.1,
        'translation_invariance': True,
    }


def test_gen_transition(common_params):
    n_layer = common_params['n_layer']
    n_child = common_params['n_child']
    p_flip = common_params['p_flip']
    variable_type = common_params['variable_type']
    flip_scale = common_params['flip_scale']
    translation_invariance = common_params['translation_invariance']
    transition, skeleton = GenTransition(n_layer, 
                                n_child, 
                                variable_type, 
                                p_flip, 
                                flip_scale, 
                                translation_invariance,
                                verbose=True)
    
    # Verify the structure of transition matrices
    for layer in range(n_layer):
        num_nodes = n_child ** layer
        num_matrices = len(transition[layer])
        
        # Check that we have the correct number of transition matrices
        assert num_matrices == num_nodes * n_child, f"Layer {layer} should have {num_nodes * n_child} matrices but has {num_matrices}"
        
        # Check dimensions of transition matrices
        assert transition[layer][0].shape == (variable_type, variable_type), f"Transition matrix shape should be ({variable_type}, {variable_type})"
        
        # Check that transition matrices are valid (rows sum to 1)
        assert np.allclose(np.sum(transition[layer][0], axis=1), 1.0), "Transition matrix rows should sum to 1"
        
        # Check skeleton matrices if translation_invariance is True
        if translation_invariance:
            assert skeleton[layer].shape == (variable_type, variable_type), f"Skeleton matrix shape should be ({variable_type}, {variable_type})"
            assert np.allclose(np.sum(skeleton[layer], axis=1), 1.0), "Skeleton matrix rows should sum to 1"

            # For each layer, check that transition matrices repeat in groups of n_child
            for i in range(0, num_matrices, n_child):
                # Only check if we have at least n_child matrices left
                if i + n_child <= num_matrices:
                    base_matrices = transition[layer][i:i+n_child]
                    
                    # Check that the next set of n_child matrices (if available) are the same
                    if i + 2*n_child <= num_matrices:
                        next_matrices = transition[layer][i+n_child:i+2*n_child]
                        for j in range(n_child):
                            assert np.allclose(base_matrices[j], next_matrices[j]), \
                                f"With translation_invariance=True, transition matrices should repeat every n_child={n_child} matrices"

def test_ghm_tree(common_params):
    n_layer = common_params['n_layer']
    n_child = common_params['n_child']
    p_flip = common_params['p_flip']
    variable_type = common_params['variable_type']
    flip_scale = common_params['flip_scale']
    translation_invariance = common_params['translation_invariance']

    transition = GenTransition(n_layer, 
                                n_child, 
                                variable_type, 
                                p_flip, 
                                flip_scale, 
                                translation_invariance,
                                verbose=False)
    
    tree = GHMTree(n_layer=n_layer, 
                 n_child=n_child, 
                 variable_type=variable_type, 
                 p_y=None,  # Will convert to tensor
                 p_flip=p_flip, 
                 transition=transition, 
                 batch_size=128, 
                 build_tree=True, 
                 root=None,
                 device='cuda')

    # Check that tree.T_value has the correct number of layers
    assert len(tree.T_value) == n_layer + 1, f"Tree should have {n_layer + 1} layers, but has {len(tree.T_value)}"
    
    # Check each layer has the correct number of items and each item has the correct batch size
    for layer in range(n_layer + 1):
        expected_items = n_child ** layer
        if layer == 0:
            expected_items = 1  # Root layer has only one node
            
        actual_items = len(tree.T_value[layer])
        assert actual_items == expected_items, f"Layer {layer} should have {expected_items} items, but has {actual_items}"
        
        # Check each item in the layer has the correct batch size
        for item_idx, item in enumerate(tree.T_value[layer]):
            assert item.size(0) == tree.batch_size, f"Item {item_idx} in layer {layer} should have batch size {tree.batch_size}, but has {item.size(0)}"
            assert item.device.type == 'cuda', f"Item {item_idx} in layer {layer} should be on cuda device, but is on {item.device.type}"