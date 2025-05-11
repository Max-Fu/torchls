import torch
import unittest

from torchls.lie_math.se3 import se3_exp_map, se3_log_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE
from torchls.variables.lie_groups import SE3Variable # For identity comparison

class TestSE3Math(unittest.TestCase):
    """Tests for SE(3) exponential and logarithmic maps."""

    def setUp(self):
        """Set up test parameters and tensors."""
        self.identity_matrix = torch.eye(4, device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_small = torch.tensor([0.01, 0.02, 0.03, 0.001, 0.002, 0.003], device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_large = torch.tensor([0.5, -0.2, 1.0, 0.1, -0.3, 0.2], device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_zero = torch.zeros(6, device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_near_pi = torch.tensor([0.1, 0.2, 0.3, torch.pi - 1e-3, 0.0002, 0.0003], device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_batch = torch.stack([self.delta_small, self.delta_large, self.delta_zero, self.delta_near_pi], dim=0)

    def test_exp_log_identity(self):
        """Test that log(exp(0)) == 0 and exp(log(I)) == I."""
        # Test log(exp(0)) == 0
        T_from_zero_delta = se3_exp_map(self.delta_zero)
        self.assertTrue(torch.allclose(T_from_zero_delta, self.identity_matrix, atol=1e-7), "exp(0) should be Identity")
        delta_recovered_from_I_exp = se3_log_map(T_from_zero_delta)
        self.assertTrue(torch.allclose(delta_recovered_from_I_exp, self.delta_zero, atol=1e-7), "log(exp(0)) should be 0")

        # Test exp(log(I)) == I (log_map of Identity should be zero vector)
        delta_from_identity = se3_log_map(self.identity_matrix)
        self.assertTrue(torch.allclose(delta_from_identity, self.delta_zero, atol=1e-7), "log(I) should be 0")
        T_recovered_from_zero_log = se3_exp_map(delta_from_identity)
        self.assertTrue(torch.allclose(T_recovered_from_zero_log, self.identity_matrix, atol=1e-7), "exp(log(I)) should be I")

    def test_exp_log_inversion_small_delta(self):
        """Test log(exp(delta)) == delta for small delta."""
        T_small = se3_exp_map(self.delta_small)
        delta_recovered_small = se3_log_map(T_small)
        self.assertTrue(torch.allclose(self.delta_small, delta_recovered_small, atol=1e-6),
                        f"SE3 exp/log inversion failed for small delta. Expected {self.delta_small}, got {delta_recovered_small}")

    def test_exp_log_inversion_large_delta(self):
        """Test log(exp(delta)) == delta for large delta."""
        T_large = se3_exp_map(self.delta_large)
        delta_recovered_large = se3_log_map(T_large)
        self.assertTrue(torch.allclose(self.delta_large, delta_recovered_large, atol=1e-6),
                        f"SE3 exp/log inversion failed for large delta. Expected {self.delta_large}, got {delta_recovered_large}")

    def test_exp_log_inversion_near_pi(self):
        """Test log(exp(delta)) == delta for delta with rotation near pi."""
        T_near_pi = se3_exp_map(self.delta_near_pi)
        delta_recovered_near_pi = se3_log_map(T_near_pi)
        # Normalization of rotation vector might lead to sign flips if theta is exactly pi
        # For near pi, it should be stable.
        self.assertTrue(torch.allclose(self.delta_near_pi, delta_recovered_near_pi, atol=1e-5),
                        f"SE3 exp/log inversion failed for delta near pi. Expected {self.delta_near_pi}, got {delta_recovered_near_pi}")

    def test_exp_map_batch(self):
        """Test batched se3_exp_map."""
        T_batch = se3_exp_map(self.delta_batch) # (B, 4, 4)
        self.assertEqual(T_batch.shape, (self.delta_batch.shape[0], 4, 4))
        for i in range(self.delta_batch.shape[0]):
            T_single = se3_exp_map(self.delta_batch[i])
            self.assertTrue(torch.allclose(T_batch[i], T_single, atol=1e-7),
                            f"Batched exp_map mismatch at index {i}")

    def test_log_map_batch(self):
        """Test batched se3_log_map."""
        T_batch_from_delta = se3_exp_map(self.delta_batch) # (B,4,4)
        recovered_delta_batch = se3_log_map(T_batch_from_delta) # (B,6)
        self.assertEqual(recovered_delta_batch.shape, self.delta_batch.shape)
        self.assertTrue(torch.allclose(recovered_delta_batch, self.delta_batch, atol=1e-5),
                        f"Batched log_map inversion failed. Expected {self.delta_batch}, got {recovered_delta_batch}")

    def test_exp_log_gradient_identity(self):
        """Test gradient of log(exp(delta)) around delta=0 is Identity."""
        delta_val = torch.zeros(6, device=DEVICE, dtype=DEFAULT_DTYPE, requires_grad=True)
        def log_exp_delta_func(d_input_func):
            # Ensure output is (6,) for Jacobian of a (6,) vector wrt (6,) vector
            return se3_log_map(se3_exp_map(d_input_func)) 

        jac = torch.autograd.functional.jacobian(log_exp_delta_func, delta_val,
                                                 strict=True, vectorize=False, create_graph=False)
        self.assertTrue(torch.allclose(jac, torch.eye(6, device=DEVICE, dtype=DEFAULT_DTYPE), atol=1e-5),
                        "Gradient of log(exp(delta)) at delta=0 should be Identity.")

    def test_exp_log_gradient_small_delta(self):
        """Test gradient of log(exp(delta)) around a small non-zero delta is Identity."""
        delta_val = self.delta_small.clone().requires_grad_(True)
        def log_exp_delta_func(d_input_func):
            return se3_log_map(se3_exp_map(d_input_func))
            
        jac = torch.autograd.functional.jacobian(log_exp_delta_func, delta_val,
                                                 strict=True, vectorize=False, create_graph=False)
        self.assertTrue(torch.allclose(jac, torch.eye(6, device=DEVICE, dtype=DEFAULT_DTYPE), atol=1e-5),
                        "Gradient of log(exp(delta)) at small delta should be Identity.")

if __name__ == '__main__':
    unittest.main() 