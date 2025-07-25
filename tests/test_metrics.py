"""
Tests for the metrics.py functions.
"""
import os
import sys

import torch
import numpy as np
from src.metrics.metrics import compute_score, mutual_knn, compute_nearest_neighbors

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create sample feature tensors for testing
        self.batch_size = 10
        self.n_layers = 3
        self.feat_dim = 16
        
        # Create normalized random feature tensors
        self.x_feats = torch.randn(self.batch_size, self.n_layers, self.feat_dim)
        self.y_feats = torch.randn(self.batch_size, self.n_layers, self.feat_dim)
        
        # Create features for knn tests
        self.feats_A = torch.randn(self.batch_size, self.feat_dim)
        self.feats_B = torch.randn(self.batch_size, self.feat_dim)
        
        # Normalize the features
        self.x_feats = self._normalize_features(self.x_feats)
        self.y_feats = self._normalize_features(self.y_feats)
        self.feats_A = torch.nn.functional.normalize(self.feats_A, p=2, dim=-1)
        self.feats_B = torch.nn.functional.normalize(self.feats_B, p=2, dim=-1)
        
    def _normalize_features(self, features):
        # Normalize features along the last dimension
        shape = features.shape
        features = features.reshape(-1, shape[-1])
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.reshape(shape)
        
    def test_compute_nearest_neighbors(self):
        # Test computing nearest neighbors
        topk = 3
        knn = compute_nearest_neighbors(self.feats_A, topk)
        
        # Check shape
        self.assertEqual(knn.shape, (self.batch_size, topk))
        
        # Check that all indices are valid
        self.assertTrue(torch.all(knn >= 0))
        self.assertTrue(torch.all(knn < self.batch_size))
        
        # Check that diagonal is not included (self is not a neighbor)
        for i in range(self.batch_size):
            self.assertNotIn(i, knn[i])
        
        print(f"Nearest neighbors test passed with topk={topk}, shape={knn.shape}")
    
    def test_mutual_knn(self):
        # Test mutual knn function
        topk = 3
        accuracy = mutual_knn(self.feats_A, self.feats_B, topk)
        
        # Check that accuracy is a float
        self.assertIsInstance(accuracy, float)
        
        # Check that accuracy is between 0 and 1
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Test with identical features (should give perfect accuracy)
        perfect_accuracy = mutual_knn(self.feats_A, self.feats_A, topk)
        self.assertAlmostEqual(perfect_accuracy, 1.0)
        
        print(f"Mutual KNN test passed with accuracy={accuracy:.4f}, perfect_accuracy={perfect_accuracy:.4f}")
    
    def test_compute_score(self):
        # Test compute_score function
        topk = 3
        
        # For now, we can test a modified version that would work correctly
        score, indices = compute_score(self.x_feats, self.y_feats, topk)
        
        # Check that score is a float
        self.assertIsInstance(score, float)
        
        # Check that score is between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Check that indices is a tuple of two integers
        self.assertIsInstance(indices, tuple)
        self.assertEqual(len(indices), 2)
        
        # Check that indices are valid layer indices
        self.assertIn(indices[0], list(range(-1, self.n_layers)))
        self.assertIn(indices[1], list(range(-1, self.n_layers)))
        
        # Test with identical features (should give higher score)
        identical_score, identical_indices = compute_score(self.x_feats, self.x_feats, topk)
        self.assertEqual(identical_score, 1.0)
        
        print(f"Compute score test passed with score={score:.4f}, best_indices={indices}")
        print(f"Identical features score={identical_score:.4f}, indices={identical_indices}")


if __name__ == "__main__":
    unittest.main() 