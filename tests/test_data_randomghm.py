"""
Tests of functions, classes, and models.
"""
import os 
import sys
src_path = os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(src_path)

import unittest 
import numpy as np
from src.data.data_random_GHM import * 

# Parameters for the tests 
n_layers = [3,4]
n_childs = [3,3]
p_ys = [np.ones(10)/10, np.ones(10)/10]
p_flips = [0.1, 0.1]
K = 4
variable_type = 10
flip_scale = 1
translation_invariance = True
sigma=0.1

def denoise_test(true_leave_values, pred_leave_values):
    """
    Test the denoising function by comparing the true and predicted leave values.
    """
    true_leave_values = np.array(true_leave_values)
    mean_power_res = np.mean(np.power(pred_leave_values,2),1)
    mean_pred_true_res = np.mean(np.multiply(pred_leave_values,true_leave_values),1)
    # print(mean_power_res.shape, mean_pred_true_res.shape)
    return np.abs(np.mean(mean_power_res) - np.mean(mean_pred_true_res))


class TestDenoising(unittest.TestCase):
    def test_conditional_denoising(self):
        batch_size = 10000
        sampler = ConditionalDenoiseSampler(n_layers, n_childs, p_ys, p_flips, flip_scale=flip_scale, sigma=sigma, translation_invariance=translation_invariance,variable_type=variable_type)
        _, res_image = sampler.get_batch(batch_size=batch_size, guide=True)
        err = denoise_test(res_image[1], res_image[-1])
        self.assertLess(err, 3e-3, "Conditional denoising failed.")
        print("Conditional denoising test passed with error: ", err)

    def test_denoising(self):
        batch_size = 10000
        sampler = DenoiseSampler(n_layers[0], n_childs[0], p_ys[0], p_flips[0],  flip_scale=flip_scale, sigma=sigma, translation_invariance=translation_invariance,variable_type=variable_type)
        res = sampler.get_batch(batch_size=batch_size, guide=True)
        err = denoise_test(res[1], res[-1])
        self.assertLess(err, 3e-3, "Denoising failed.")
        print("Denoising test passed with error: ", err)

