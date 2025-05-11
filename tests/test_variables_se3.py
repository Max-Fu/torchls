import torch
import unittest

from torchls.variables.base import Variable # To reset ID counter
from torchls.variables.lie_groups import SE3Variable
from torchls.lie_math.se3 import se3_exp_map, se3_log_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

class TestSE3Variable(unittest.TestCase):
    """Tests for the SE3Variable class operations."""

    def setUp(self):
        """Reset Variable ID counter and set up common tensors."""
        Variable._next_id = 0 # Reset unique ID counter for fresh state in each test
        self.identity_mat_4x4 = torch.eye(4, device=DEVICE, dtype=DEFAULT_DTYPE)
        self.identity_batch_1x4x4 = self.identity_mat_4x4.unsqueeze(0)
        
        self.delta_vec_6 = torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], device=DEVICE, dtype=DEFAULT_DTYPE)
        self.delta_batch_2x6 = torch.stack([
            self.delta_vec_6,
            torch.tensor([-0.1, -0.2, -0.3, -0.01, -0.02, -0.03], device=DEVICE, dtype=DEFAULT_DTYPE)
        ])
        self.T1_4x4 = se3_exp_map(self.delta_vec_6) # (1,4,4) - se3_exp_map returns batched
        self.T2_4x4 = se3_exp_map(self.delta_batch_2x6[1]) # (1,4,4) - se3_exp_map returns batched

    def test_initialization(self):
        """Test SE3Variable initialization with and without initial values."""
        # Default initialization (should be identity)
        var_default = SE3Variable(name="DefaultSE3")
        self.assertEqual(var_default.name, "DefaultSE3") # Name includes auto-ID
        self.assertTrue(torch.allclose(var_default.initial_value, self.identity_batch_1x4x4, atol=1e-7),
                        "Default initial_value should be batched identity (1,4,4).")
        self.assertEqual(var_default.tangent_dim, 6)

        # Initialization with a (4,4) tensor
        # self.T1_4x4 is (1,4,4), so squeeze it to get a (4,4) for this test case
        var_4x4 = SE3Variable(initial_value=self.T1_4x4.squeeze(0), name="GivenSE3_4x4")
        # Expect it to be stored as (1,4,4), which self.T1_4x4 already is.
        self.assertTrue(torch.allclose(var_4x4.initial_value, self.T1_4x4, atol=1e-7),
                        "Initial_value from (4,4) should be stored as (1,4,4).")

        # Initialization with a (B,4,4) tensor
        T_batch_2x4x4 = se3_exp_map(self.delta_batch_2x6) # (2,4,4)
        var_Bx4x4 = SE3Variable(initial_value=T_batch_2x4x4, name="GivenSE3_Bx4x4")
        self.assertTrue(torch.allclose(var_Bx4x4.initial_value, T_batch_2x4x4, atol=1e-7),
                        "Initial_value from (B,4,4) should be stored as is.")
        
        # Test assertion for wrong initial_value shape
        with self.assertRaises(AssertionError):
            SE3Variable(initial_value=torch.randn(3,3, device=DEVICE, dtype=DEFAULT_DTYPE))
        with self.assertRaises(AssertionError):
            SE3Variable(initial_value=torch.randn(1,3,4, device=DEVICE, dtype=DEFAULT_DTYPE))

    def test_identity_classmethod(self):
        """Test the SE3Variable.identity() class method for various batch sizes."""
        # No batch_size (should be (1,4,4))
        id_default_batch = SE3Variable.identity()
        self.assertEqual(id_default_batch.shape, (1,4,4))
        self.assertTrue(torch.allclose(id_default_batch.squeeze(0), self.identity_mat_4x4, atol=1e-7))

        # batch_size = () (should be (4,4))
        id_unbatched = SE3Variable.identity(batch_size=())
        self.assertEqual(id_unbatched.shape, (4,4))
        self.assertTrue(torch.allclose(id_unbatched, self.identity_mat_4x4, atol=1e-7))

        # batch_size = 0 (should be (4,4))
        id_zero_batch = SE3Variable.identity(batch_size=0)
        self.assertEqual(id_zero_batch.shape, (4,4))
        self.assertTrue(torch.allclose(id_zero_batch, self.identity_mat_4x4, atol=1e-7))

        # batch_size = N (should be (N,4,4))
        N = 3
        id_N_batch = SE3Variable.identity(batch_size=N)
        self.assertEqual(id_N_batch.shape, (N,4,4))
        for n_idx in range(N):
            self.assertTrue(torch.allclose(id_N_batch[n_idx], self.identity_mat_4x4, atol=1e-7))

    def test_retract(self):
        """Test the retract method for SE3Variable."""
        var = SE3Variable() # Initialized to identity (1,4,4)
        
        # Case 1: current_value (1,4,4), delta (6,)
        val_current_1x4x4 = var.initial_value # (1,4,4)
        delta_6 = self.delta_vec_6 # (6,)
        val_retracted_1 = var.retract(val_current_1x4x4, delta_6)
        expected_T1 = se3_exp_map(delta_6) @ val_current_1x4x4.squeeze(0)
        self.assertEqual(val_retracted_1.shape, (1,4,4))
        self.assertTrue(torch.allclose(val_retracted_1.squeeze(0), expected_T1, atol=1e-6))

        # Case 2: current_value (4,4), delta (6,)
        val_current_4x4 = self.identity_mat_4x4 # (4,4)
        val_retracted_2 = var.retract(val_current_4x4, delta_6)
        expected_T2 = se3_exp_map(delta_6) @ val_current_4x4
        self.assertEqual(val_retracted_2.shape, (4,4))
        self.assertTrue(torch.allclose(val_retracted_2, expected_T2, atol=1e-6))

        # Case 3: current_value (B,4,4), delta (B,6)
        T_batch_B44 = se3_exp_map(self.delta_batch_2x6) # (2,4,4)
        delta_batch_B6 = self.delta_batch_2x6 # (2,6)
        val_retracted_3 = var.retract(T_batch_B44, delta_batch_B6)
        expected_T3_0 = se3_exp_map(delta_batch_B6[0]) @ T_batch_B44[0]
        expected_T3_1 = se3_exp_map(delta_batch_B6[1]) @ T_batch_B44[1]
        self.assertEqual(val_retracted_3.shape, (2,4,4))
        self.assertTrue(torch.allclose(val_retracted_3[0], expected_T3_0, atol=1e-6))
        self.assertTrue(torch.allclose(val_retracted_3[1], expected_T3_1, atol=1e-6))

        # Case 4: current_value (1,4,4), delta (B,6) -> broadcast current_value
        val_retracted_4 = var.retract(val_current_1x4x4, delta_batch_B6)
        expected_T4_0 = se3_exp_map(delta_batch_B6[0]) @ val_current_1x4x4.squeeze(0)
        expected_T4_1 = se3_exp_map(delta_batch_B6[1]) @ val_current_1x4x4.squeeze(0)
        self.assertEqual(val_retracted_4.shape, (2,4,4))
        self.assertTrue(torch.allclose(val_retracted_4[0], expected_T4_0, atol=1e-6))
        self.assertTrue(torch.allclose(val_retracted_4[1], expected_T4_1, atol=1e-6))
        
        # Case 5: current_value (B,4,4), delta (6,) -> broadcast delta
        val_retracted_5 = var.retract(T_batch_B44, delta_6)
        expected_T5_0 = se3_exp_map(delta_6) @ T_batch_B44[0]
        expected_T5_1 = se3_exp_map(delta_6) @ T_batch_B44[1]
        self.assertEqual(val_retracted_5.shape, (2,4,4))
        self.assertTrue(torch.allclose(val_retracted_5[0], expected_T5_0, atol=1e-6))
        self.assertTrue(torch.allclose(val_retracted_5[1], expected_T5_1, atol=1e-6))

        # Case 6: Incompatible batch sizes
        with self.assertRaises(ValueError):
            incompat_delta = torch.randn(3, 6, device=DEVICE, dtype=DEFAULT_DTYPE)
            var.retract(T_batch_B44, incompat_delta) # (2,4,4) and (3,6)

    def test_local_coordinates(self):
        """Test the local_coordinates method for SE3Variable."""
        var = SE3Variable() # Not strictly needed other than for calling the method

        # Case 1: value1 (4,4), value2 (4,4)
        T1_val1 = self.identity_mat_4x4
        T2_val2 = self.T1_4x4 # exp(delta_vec_6), which is (1,4,4)
        delta_lc_1 = var.local_coordinates(T1_val1, T2_val2)
        self.assertEqual(delta_lc_1.shape, (1, 6)) # Expect (1,6) due to broadcasting/promotion
        self.assertTrue(torch.allclose(delta_lc_1.squeeze(0), self.delta_vec_6, atol=1e-6)) # Compare value

        # Case 2: value1 (1,4,4), value2 (1,4,4)
        T1_val1_b1 = self.identity_batch_1x4x4
        T2_val2_b1 = self.T1_4x4 # This is already (1,4,4)
        delta_lc_2 = var.local_coordinates(T1_val1_b1, T2_val2_b1)
        self.assertEqual(delta_lc_2.shape, (1, 6))
        self.assertTrue(torch.allclose(delta_lc_2.squeeze(0), self.delta_vec_6, atol=1e-6))

        # Case 3: value1 (B,4,4), value2 (B,4,4)
        batch_T1 = SE3Variable.identity(batch_size=2) # Batch of 2 identities (2,4,4)
        batch_T2 = se3_exp_map(self.delta_batch_2x6)    # Batch of 2 transforms (2,4,4)
        delta_lc_3 = var.local_coordinates(batch_T1, batch_T2)
        self.assertEqual(delta_lc_3.shape, (2,6))
        self.assertTrue(torch.allclose(delta_lc_3, self.delta_batch_2x6, atol=1e-6))

        # Case 4: value1 (1,4,4), value2 (B,4,4) -> broadcast value1
        delta_lc_4 = var.local_coordinates(self.identity_batch_1x4x4, batch_T2)
        self.assertEqual(delta_lc_4.shape, (2,6))
        self.assertTrue(torch.allclose(delta_lc_4, self.delta_batch_2x6, atol=1e-6))

        # Case 5: value1 (B,4,4), value2 (1,4,4) -> broadcast value2
        # T2 here is exp(delta_vec_6). So we expect local_coords(batch_T1_identities, T2) to be delta_vec_6 for each row
        T2_single_batch = self.T1_4x4 # This is already (1,4,4)
        delta_lc_5 = var.local_coordinates(batch_T1, T2_single_batch)
        self.assertEqual(delta_lc_5.shape, (2,6))
        self.assertTrue(torch.allclose(delta_lc_5[0], self.delta_vec_6, atol=1e-6))
        self.assertTrue(torch.allclose(delta_lc_5[1], self.delta_vec_6, atol=1e-6))
        
        # Case 6: Incompatible batch sizes
        with self.assertRaises(ValueError):
            incompat_T = se3_exp_map(torch.randn(3, 6, device=DEVICE, dtype=DEFAULT_DTYPE))
            var.local_coordinates(batch_T1, incompat_T) # (2,4,4) and (3,4,4)

    def test_retract_local_coordinates_inversion(self):
        """Test that local_coordinates(T, retract(T, delta)) == delta."""
        var = SE3Variable()
        T_current_val_unbatched = self.T1_4x4 # (4,4)
        T_current_val_batched = self.T1_4x4.unsqueeze(0) # (1,4,4)
        delta_update_unbatched = self.delta_vec_6 # (6,)
        delta_update_batched = self.delta_batch_2x6 # (2,6)

        # Unbatched current, Unbatched delta
        T_retracted_1 = var.retract(T_current_val_unbatched, delta_update_unbatched)
        delta_recovered_1 = var.local_coordinates(T_current_val_unbatched, T_retracted_1)
        self.assertTrue(torch.allclose(delta_update_unbatched, delta_recovered_1, atol=1e-6))

        # Batched current (1,4,4), Unbatched delta (6,)
        T_retracted_2 = var.retract(T_current_val_batched, delta_update_unbatched)
        delta_recovered_2 = var.local_coordinates(T_current_val_batched, T_retracted_2)
        self.assertTrue(torch.allclose(delta_update_unbatched, delta_recovered_2.squeeze(0), atol=1e-6))

        # Batched current (B,4,4), Batched delta (B,6) -> using T_batch_B44 from retract test
        T_batch_B44 = se3_exp_map(torch.randn(2,6, device=DEVICE, dtype=DEFAULT_DTYPE)) # Start with some batch
        T_retracted_3 = var.retract(T_batch_B44, delta_update_batched)
        delta_recovered_3 = var.local_coordinates(T_batch_B44, T_retracted_3)
        # import pdb; pdb.set_trace()
        self.assertTrue(torch.allclose(delta_update_batched, delta_recovered_3, atol=1e-6))

    def test_inverse_method(self):
        """Test the SE3Variable.inverse() method."""
        var = SE3Variable() # Helper instance

        # 1. Inverse of identity (4,4)
        T_identity_4x4 = self.identity_mat_4x4
        inv_T_identity_4x4 = var.inverse(T_identity_4x4)
        self.assertEqual(inv_T_identity_4x4.shape, (4,4))
        self.assertTrue(torch.allclose(inv_T_identity_4x4, self.identity_mat_4x4, atol=1e-7))

        # 2. Inverse of identity (1,4,4)
        T_identity_1x4x4 = self.identity_batch_1x4x4
        inv_T_identity_1x4x4 = var.inverse(T_identity_1x4x4)
        self.assertEqual(inv_T_identity_1x4x4.shape, (1,4,4))
        self.assertTrue(torch.allclose(inv_T_identity_1x4x4, self.identity_batch_1x4x4, atol=1e-7))

        # 3. Inverse of a single non-identity SE(3) matrix (input 1,4,4, check result is 1,4,4)
        # self.T1_4x4 is already (1,4,4)
        T_known_1x4x4 = self.T1_4x4 
        inv_T_known_1x4x4 = var.inverse(T_known_1x4x4)
        # Check T @ T_inv = Identity
        recomposed_identity_1 = torch.matmul(T_known_1x4x4, inv_T_known_1x4x4)
        self.assertTrue(torch.allclose(recomposed_identity_1, self.identity_batch_1x4x4, atol=1e-6))
        self.assertEqual(inv_T_known_1x4x4.shape, T_known_1x4x4.shape)

        # 4. Inverse of a single non-identity SE(3) matrix (input 4,4, check result is 4,4)
        T_known_4x4 = self.T1_4x4.squeeze(0)
        inv_T_known_4x4 = var.inverse(T_known_4x4)
        recomposed_identity_2 = torch.matmul(T_known_4x4, inv_T_known_4x4)
        self.assertTrue(torch.allclose(recomposed_identity_2, self.identity_mat_4x4, atol=1e-6))
        self.assertEqual(inv_T_known_4x4.shape, T_known_4x4.shape)

        # 5. Inverse of a batch of SE(3) matrices (B,4,4)
        T_batch_B44 = se3_exp_map(self.delta_batch_2x6) # (2,4,4)
        inv_T_batch_B44 = var.inverse(T_batch_B44)
        recomposed_identity_batch = torch.matmul(T_batch_B44, inv_T_batch_B44)
        expected_identity_batch = SE3Variable.identity(batch_size=T_batch_B44.shape[0]) # (2,4,4) of identities
        self.assertTrue(torch.allclose(recomposed_identity_batch, expected_identity_batch, atol=1e-6))
        self.assertEqual(inv_T_batch_B44.shape, T_batch_B44.shape)

        # 6. Test inverse(inverse(T)) = T
        inv_inv_T_known_1x4x4 = var.inverse(inv_T_known_1x4x4)
        self.assertTrue(torch.allclose(inv_inv_T_known_1x4x4, T_known_1x4x4, atol=1e-6))

        inv_inv_T_batch_B44 = var.inverse(inv_T_batch_B44)
        self.assertTrue(torch.allclose(inv_inv_T_batch_B44, T_batch_B44, atol=1e-6))

    def test_compose_method(self):
        """Test the SE3Variable.compose() method."""
        var = SE3Variable() # Helper instance

        T_id_1x4x4 = self.identity_batch_1x4x4
        T_id_4x4 = self.identity_mat_4x4
        
        # self.T1_4x4 is (1,4,4), self.T2_4x4 is (1,4,4)
        T1_val = self.T1_4x4 
        T2_val = self.T2_4x4 

        # 1. Compose T1 with T2 (both 1,4,4)
        composed_T1_T2 = var.compose(T1_val, T2_val)
        expected_T1_T2 = torch.matmul(T1_val, T2_val)
        self.assertTrue(torch.allclose(composed_T1_T2, expected_T1_T2, atol=1e-7))
        self.assertEqual(composed_T1_T2.shape, (1,4,4))

        # 2. Compose T1 (1,4,4) with T_id (1,4,4)
        composed_T1_Id = var.compose(T1_val, T_id_1x4x4)
        self.assertTrue(torch.allclose(composed_T1_Id, T1_val, atol=1e-7))

        # 3. Compose T_id (1,4,4) with T1 (1,4,4)
        composed_Id_T1 = var.compose(T_id_1x4x4, T1_val)
        self.assertTrue(torch.allclose(composed_Id_T1, T1_val, atol=1e-7))

        # 4. Compose T1 (squeezed to 4,4) with T2 (squeezed to 4,4)
        T1_val_4x4 = T1_val.squeeze(0)
        T2_val_4x4 = T2_val.squeeze(0)
        composed_4x4 = var.compose(T1_val_4x4, T2_val_4x4)
        expected_4x4 = torch.matmul(T1_val_4x4, T2_val_4x4)
        self.assertTrue(torch.allclose(composed_4x4, expected_4x4, atol=1e-7))
        self.assertEqual(composed_4x4.shape, (4,4))

        # 5. Compose T_id (4,4) with T1 (squeezed to 4,4)
        composed_Id4x4_T1_4x4 = var.compose(T_id_4x4, T1_val_4x4)
        self.assertTrue(torch.allclose(composed_Id4x4_T1_4x4, T1_val_4x4, atol=1e-7))

        # 6. Batched composition
        # T_batch_A: (2,4,4), T_batch_B: (2,4,4)
        T_batch_A = se3_exp_map(self.delta_batch_2x6) # (2,4,4)
        delta_batch_alt = torch.stack([
            torch.tensor([0.01, 0.02, 0.03, 0.1, 0.2, 0.3], device=DEVICE, dtype=DEFAULT_DTYPE),
            torch.tensor([-0.03, -0.01, -0.02, -0.3, -0.1, -0.2], device=DEVICE, dtype=DEFAULT_DTYPE)
        ])
        T_batch_B = se3_exp_map(delta_batch_alt) # (2,4,4)
        composed_batch = var.compose(T_batch_A, T_batch_B)
        expected_batch_composed = torch.matmul(T_batch_A, T_batch_B)
        self.assertTrue(torch.allclose(composed_batch, expected_batch_composed, atol=1e-7))
        self.assertEqual(composed_batch.shape, (2,4,4))

        # 7. Broadcasting: T1_val (1,4,4) and T_batch_B (2,4,4)
        # torch.matmul will broadcast T1_val to (2,4,4)
        composed_broadcast1 = var.compose(T1_val, T_batch_B)
        expected_broadcast1 = torch.matmul(T1_val, T_batch_B)
        self.assertTrue(torch.allclose(composed_broadcast1, expected_broadcast1, atol=1e-7))
        self.assertEqual(composed_broadcast1.shape, (2,4,4))

        # 8. Broadcasting: T_batch_A (2,4,4) and T2_val (1,4,4)
        composed_broadcast2 = var.compose(T_batch_A, T2_val)
        expected_broadcast2 = torch.matmul(T_batch_A, T2_val)
        self.assertTrue(torch.allclose(composed_broadcast2, expected_broadcast2, atol=1e-7))
        self.assertEqual(composed_broadcast2.shape, (2,4,4))

if __name__ == '__main__':
    unittest.main() 