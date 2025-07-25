from ghmclip.data.data_random_GHM import * 
from ghmclip.models.functional import * 
import pytest 


@pytest.fixture
def common_params():
    return {
        'n_layer': 3,
        'n_child': 2,
        'p_y': torch.ones(5)/5,
        'variable_type': 5,
        'p_flip': 0.2,
        'flip_scale': 1,
        'sigma': 1,
        'translation_invariance': True,
        'batch_size': 500000,
        'seedtree': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def test_classification_sampler(common_params):
    # Extract only the required parameters
    params = common_params
    # ClassificationSampler has some positional args, so we need to handle them separately
    sampler = SingleSampler(
        n_layer=params['n_layer'], 
        n_child=params['n_child'], 
        p_y=params['p_y'], 
        p_flip=params['p_flip'], 
        variable_type=params['variable_type'], 
        translation_invariance=params['translation_invariance'],
        seedtree=params['seedtree'],
        device=params['device']
    )
    
    # Test the get_batch function
    leave_values, root_values, guided_info, posterior_probability_CLS = sampler.get_batch(batch_size=common_params['batch_size'], guide=True, job='clas')

    # Test the shape of the output
    n_leaves = params['n_child']**params['n_layer']
    assert leave_values.shape == (common_params['batch_size'], n_leaves)
    assert root_values.shape == (common_params['batch_size'],)
    assert posterior_probability_CLS.shape == (common_params['batch_size'], params['variable_type'])

    # Test that each row of posterior probability sums to 1
    row_sums = torch.sum(posterior_probability_CLS, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Posterior probabilities should sum to 1 for each sample"

    # Test the posterior probability is correct 
    root_values = root_values.float()
    possible_root_values = torch.tensor(range(params['variable_type'])).view(-1, 1).to(params['device'])
    print("possible_root_values", possible_root_values)
    root_mean = posterior_probability_CLS@ possible_root_values.float()

    print("predict_mean", root_mean.mean(), "real_mean", root_values.mean())
    print(posterior_probability_CLS[:10,:], root_values[:10])
    assert torch.abs(torch.mean(root_mean) - torch.mean(root_values)) < 1e-2
    power_mean = root_mean.flatten() * root_values.flatten()
    real_power = torch.pow(root_mean, 2)
    print(power_mean[:10], real_power[:10])
    print("predict_power", power_mean.mean(), "real_power", real_power.mean())
    assert torch.abs(power_mean.mean() - real_power.mean()) < 1e-2

def test_denoise_sampler(common_params):
    # Filter only the parameters needed for DenoiseSampler
    params = common_params
    
    # Use dictionary unpacking for all parameters
    sampler = SingleSampler(n_layer=params['n_layer'], 
                            n_child=params['n_child'], 
                            p_y=params['p_y'], 
                            p_flip=params['p_flip'], 
                            sigma=params['sigma'],
                            flip_scale=params['flip_scale'],
                            variable_type=params['variable_type'], 
                            translation_invariance=params['translation_invariance'],
                            seedtree=params['seedtree'],
                            device=params['device'])
    
    zs, xs, guided_info, posterior_mean_DNS = sampler.get_batch(batch_size=common_params['batch_size'], guide=True, job='dns')

    # Test the shape of the output
    n_leaves = params['n_child']**params['n_layer']
    assert zs.shape == (common_params['batch_size'], n_leaves)
    assert xs.shape == (common_params['batch_size'], n_leaves)
    assert posterior_mean_DNS.shape == (common_params['batch_size'], n_leaves)
    
    # Check that tensors are on the correct device
    assert zs.device.type == params['device']
    assert xs.device.type == params['device']
    assert posterior_mean_DNS.device.type == params['device']

    # Test the posterior probability is correct  
    ## Test mean 
    mean_xs = posterior_mean_DNS.mean(dim=0)
    mean_true = xs.float().mean(dim=0)
    assert torch.linalg.norm(mean_xs - mean_true) < 1e-2
    
    ## Test power 
    power_xs = posterior_mean_DNS * xs 
    power_pos = torch.pow(posterior_mean_DNS, 2) 
    print("power_xs", power_xs.mean(dim=0), "power_pos", power_pos[:5,:])
    assert torch.linalg.norm(power_xs.mean(dim=0) - power_pos.mean(dim=0)) < 1e-2

    # Test the corresponding indices 
    indices = gen_guide_indices(job='dns', n_tree_layer=params['n_layer'], n_class=params['variable_type'])

    assert len(indices) == len(guided_info) 
    for i in range(len(indices)):
        assert guided_info[i].shape[-1] == indices[i].shape[0]


def test_nwp_sampler(common_params):
    params = common_params
    n_token = params['n_child']**params['n_layer']
    sampler = SingleSampler(n_layer=params['n_layer'], 
                        n_child=params['n_child'], 
                        p_y=params['p_y'], 
                        p_flip=params['p_flip'], 
                        variable_type=params['variable_type'], 
                        translation_invariance=params['translation_invariance'], 
                        seedtree=params['seedtree'], 
                        device=params['device'])
    
    inputs, targets, guided_info, posterior_probability_NWP = sampler.get_batch(batch_size=common_params['batch_size'], guide=True, job='nwp')

    # Test the shape of the output
    assert inputs.shape == (common_params['batch_size'], n_token-1)
    assert targets.shape == (common_params['batch_size'], n_token-1)
    assert posterior_probability_NWP.shape == (common_params['batch_size'], n_token-1, params['variable_type']) 

    # Test the devices are correct 
    assert inputs.device.type == params['device']
    assert targets.device.type == params['device']
    assert posterior_probability_NWP.device.type == params['device']

    # Test the posterior probability is correct 
    ## Test mean 
    mean_true = targets.float().mean(dim=0)
    possible_values = torch.tensor(range(params['variable_type'])).view(-1, 1).to(params['device']).float()
    pred = (posterior_probability_NWP@possible_values)[:,:,0]
    mean_pred = pred.mean(dim=0)
    print("mean_pred", mean_pred, "mean_true", mean_true)
    print("pred.shape, targets.shape", pred.shape, targets.shape)
    assert torch.linalg.norm(mean_pred - mean_true) < 1e-2

    ## Test power
    power_pred = pred * targets
    power_pos = torch.pow(pred, 2)
    print("power_pred", power_pred.mean(dim=0), "power_pos", power_pos.mean(dim=0))
    assert torch.linalg.norm(power_pred.mean(dim=0) - power_pos.mean(dim=0)) < 1e-2

    # Test the corresponding indices 
    indices = gen_guide_indices(job='nwp', n_tree_layer=params['n_layer'], n_class=params['variable_type'])

    assert len(indices) == len(guided_info) 
    for i in range(len(indices)):
        assert guided_info[i].shape[-1] == indices[i].shape[0]
    
    
    
    
    