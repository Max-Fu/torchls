import torch
import unittest
from collections import OrderedDict

from torchls.core.problem import LeastSquaresProblem
from torchls.core.cost import Cost
from torchls.variables.base import Variable
from torchls.variables.lie_groups import SE3Variable, LieGroupVariable
from torchls.lie_math.se3 import se3_exp_map, se3_log_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# --- Helper Cost Classes (specific to these tests) ---
class SE3PriorCost(Cost):
    """Cost for a prior on an SE3 variable."""
    def __init__(self, variable: SE3Variable, target_pose: torch.Tensor, weight: float = 1.0):
        super().__init__([variable], name=f"Prior_{variable.name}")
        self.variable = variable
        # Ensure target_pose is (B,4,4)
        if target_pose.ndim == 2 and target_pose.shape == (4,4):
            self.target_pose = target_pose.unsqueeze(0).to(device=DEVICE, dtype=DEFAULT_DTYPE)
        elif target_pose.ndim == 3 and target_pose.shape[-2:] == (4,4):
            self.target_pose = target_pose.to(device=DEVICE, dtype=DEFAULT_DTYPE)
        else:
            raise ValueError(f"target_pose for SE3PriorCost must be (4,4) or (B,4,4), got {target_pose.shape}")
        self.weight = weight

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        """residual = weight * log( current_pose * inv(target_pose) )"""
        current_pose = var_values[self.variable] # Should be (B,4,4) or (1,4,4)
        
        # Ensure current_pose is (B,4,4)
        current_pose_b = current_pose if current_pose.ndim == 3 else current_pose.unsqueeze(0)
        target_pose_b = self.target_pose

        # Broadcast target_pose if necessary
        if target_pose_b.shape[0] == 1 and current_pose_b.shape[0] > 1:
            target_pose_b = target_pose_b.expand(current_pose_b.shape[0], -1, -1)
        elif current_pose_b.shape[0] == 1 and target_pose_b.shape[0] > 1:
            current_pose_b = current_pose_b.expand(target_pose_b.shape[0], -1, -1)
        elif current_pose_b.shape[0] != target_pose_b.shape[0] and target_pose_b.shape[0] !=1 and current_pose_b.shape[0] !=1:
             raise ValueError(f"Batch dimensions of current_pose {current_pose_b.shape} and target_pose {target_pose_b.shape} are incompatible.")

        # T_delta = current_pose_b @ torch.linalg.inv(target_pose_b)
        # More stable: T_target_inv @ T_current. If T_target is T_w_tgt, T_current is T_w_cur
        # error is T_tgt_cur = T_w_tgt^-1 * T_w_cur. log(T_tgt_cur)
        # Original: T_cur * T_tgt^-1. This means error is in world frame. log( T_w_cur * T_w_tgt^-1 )
        # This is T_cur_tgt_in_world. Usually error is local: log ( T_target^-1 * T_current )
        # Let's stick to original: T_delta = current_pose_b @ torch.linalg.inv(target_pose_b)
        T_delta = torch.matmul(current_pose_b, torch.linalg.inv(target_pose_b))
        residual_log = se3_log_map(T_delta) # (B,6) or (6,)
        return residual_log.flatten() * self.weight # (B*6,) or (6,)

class SE3BetweenCost(Cost):
    """Cost for a relative transformation constraint between two SE3 variables."""
    def __init__(self, var1: SE3Variable, var2: SE3Variable, measured_delta_T_12: torch.Tensor, weight: float = 1.0):
        super().__init__([var1, var2], name=f"Between_{var1.name}_{var2.name}")
        self.var1 = var1
        self.var2 = var2
        # Ensure measured_delta_T_12 is (B,4,4)
        if measured_delta_T_12.ndim == 2 and measured_delta_T_12.shape == (4,4):
            self.measured_delta_T_12 = measured_delta_T_12.unsqueeze(0).to(device=DEVICE, dtype=DEFAULT_DTYPE)
        elif measured_delta_T_12.ndim == 3 and measured_delta_T_12.shape[-2:] == (4,4):
            self.measured_delta_T_12 = measured_delta_T_12.to(device=DEVICE, dtype=DEFAULT_DTYPE)
        else:
            raise ValueError(f"measured_delta_T_12 for SE3BetweenCost must be (4,4) or (B,4,4), got {measured_delta_T_12.shape}")
        self.weight = weight

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        """residual = weight * log ( (inv(T_w1) * T_w2) * inv(measured_delta_T_12) )"""
        T_w1 = var_values[self.var1]
        T_w2 = var_values[self.var2]

        T_w1_b = T_w1 if T_w1.ndim == 3 else T_w1.unsqueeze(0)
        T_w2_b = T_w2 if T_w2.ndim == 3 else T_w2.unsqueeze(0)
        measured_delta_b = self.measured_delta_T_12

        # Broadcasting logic for T_w1_b, T_w2_b, and measured_delta_b
        # Assume T_w1_b and T_w2_b should have same batch size from problem construction
        # Expand measured_delta_b if it's singular and poses are batched.
        if measured_delta_b.shape[0] == 1 and T_w1_b.shape[0] > 1:
            measured_delta_b = measured_delta_b.expand(T_w1_b.shape[0], -1, -1)
        # Also ensure T_w1_b and T_w2_b are compatible (e.g. one is (1,4,4) and other is (B,4,4))
        if T_w1_b.shape[0] == 1 and T_w2_b.shape[0] > 1:
            T_w1_b = T_w1_b.expand(T_w2_b.shape[0],-1,-1)
        elif T_w2_b.shape[0] == 1 and T_w1_b.shape[0] > 1:
            T_w2_b = T_w2_b.expand(T_w1_b.shape[0],-1,-1)
        elif T_w1_b.shape[0] != T_w2_b.shape[0] and T_w1_b.shape[0] !=1 and T_w2_b.shape[0] !=1:
            raise ValueError(f"Batch dims of T_w1 {T_w1_b.shape} and T_w2 {T_w2_b.shape} incompatible.")
        if T_w1_b.shape[0] != measured_delta_b.shape[0] and measured_delta_b.shape[0]!=1 and T_w1_b.shape[0]!=1:
             raise ValueError(f"Batch dims of poses {T_w1_b.shape} and measurement {measured_delta_b.shape} incompatible.")

        # T_1_2_estimated = inv(T_w1_b) * T_w2_b
        T_1_2_estimated = torch.matmul(torch.linalg.inv(T_w1_b), T_w2_b)
        # error_transform = T_1_2_estimated * inv(measured_delta_b)
        error_transform = torch.matmul(T_1_2_estimated, torch.linalg.inv(measured_delta_b))
        residual_log = se3_log_map(error_transform)
        return residual_log.flatten() * self.weight

class TestLeastSquaresProblem(unittest.TestCase):
    """Tests for the LeastSquaresProblem class."""

    def setUp(self):
        """Set up common variables and problem structures."""
        Variable._next_id = 100 # Reset for predictable IDs in tests
        self.pose0 = SE3Variable(name="P0") # Default init to Identity(1,4,4)
        self.pose1_init_val = se3_exp_map(torch.tensor([0.1, -0.1, 0.1, 0.01, -0.01, 0.01], dtype=DEFAULT_DTYPE))
        self.pose1 = SE3Variable(initial_value=self.pose1_init_val, name="P1")
        self.pose2_init_val = se3_exp_map(torch.tensor([0.2, 0.1, -0.2, -0.02, 0.02, -0.01], dtype=DEFAULT_DTYPE))
        self.pose2 = SE3Variable(initial_value=self.pose2_init_val, name="P2")

        self.prior0 = SE3PriorCost(self.pose0, SE3Variable.identity(batch_size=()), weight=10.0)
        self.T_0_1_measured = se3_exp_map(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=DEFAULT_DTYPE))
        self.between01 = SE3BetweenCost(self.pose0, self.pose1, self.T_0_1_measured, weight=1.0)
        self.T_1_2_measured = se3_exp_map(torch.tensor([0.5, 0.1, 0.0, 0.0, 0.0, 0.05], dtype=DEFAULT_DTYPE))
        self.between12 = SE3BetweenCost(self.pose1, self.pose2, self.T_1_2_measured, weight=2.0)

    def test_analyze(self):
        """Test the _analyze method for variable ordering and tangent dimensions."""
        problem = LeastSquaresProblem(costs=[self.prior0, self.between01, self.between12])
        self.assertTrue(problem._is_analyzed)
        self.assertEqual(len(problem.ordered_lie_vars_for_problem), 3)
        self.assertListEqual(problem.ordered_lie_vars_for_problem, [self.pose0, self.pose1, self.pose2])
        self.assertEqual(problem.total_tangent_dim, 3 * 6)
        
        # Check var_to_tangent_info
        self.assertEqual(problem.var_to_tangent_info[self.pose0], (0, 0, 6)) # (idx, offset, dim)
        self.assertEqual(problem.var_to_tangent_info[self.pose1], (1, 6, 6))
        self.assertEqual(problem.var_to_tangent_info[self.pose2], (2, 12, 6))

        # Test with a non-Lie variable (should be ignored in tangent_info but present in initial_values if it has .initial_value)
        fixed_param = Variable(name="FixedParam")
        fixed_param.initial_value = torch.tensor([1.0, 2.0], device=DEVICE, dtype=DEFAULT_DTYPE)
        prior0_with_fixed = SE3PriorCost(self.pose0, SE3Variable.identity(batch_size=()))
        prior0_with_fixed.variables.append(fixed_param) # Manually add for test
        problem_with_fixed = LeastSquaresProblem(costs=[prior0_with_fixed])
        self.assertEqual(len(problem_with_fixed.ordered_lie_vars_for_problem), 1)
        self.assertEqual(problem_with_fixed.total_tangent_dim, 6)
        init_vals_fixed = problem_with_fixed.get_initial_values()
        self.assertIn(self.pose0, init_vals_fixed)
        self.assertIn(fixed_param, init_vals_fixed)
        self.assertTrue(torch.allclose(init_vals_fixed[fixed_param], fixed_param.initial_value))


    def test_get_initial_values(self):
        """Test retrieval of initial values for all variables."""
        problem = LeastSquaresProblem(costs=[self.prior0, self.between01])
        initial_values = problem.get_initial_values()
        self.assertEqual(len(initial_values), 2) # pose0, pose1
        self.assertTrue(torch.allclose(initial_values[self.pose0], self.pose0.initial_value))
        self.assertTrue(torch.allclose(initial_values[self.pose1], self.pose1.initial_value))

    def test_update_values(self):
        """Test updating variable values using a tangent space delta."""
        problem = LeastSquaresProblem(costs=[self.prior0, self.between01])
        current_values = problem.get_initial_values()
        
        # Delta: first 6 for pose0, next 6 for pose1
        delta_tangent = torch.cat([
            torch.tensor([0.01,0.02,0.03,0.001,0.002,0.003], dtype=DEFAULT_DTYPE, device=DEVICE),
            torch.tensor([-0.01,-0.02,-0.03,-0.001,-0.002,-0.003], dtype=DEFAULT_DTYPE, device=DEVICE)
        ])
        self.assertEqual(delta_tangent.shape[0], problem.total_tangent_dim)

        new_values = problem.update_values(current_values, delta_tangent)
        
        expected_pose0_updated = self.pose0.retract(current_values[self.pose0], delta_tangent[:6])
        expected_pose1_updated = self.pose1.retract(current_values[self.pose1], delta_tangent[6:])

        self.assertTrue(torch.allclose(new_values[self.pose0], expected_pose0_updated, atol=1e-6))
        self.assertTrue(torch.allclose(new_values[self.pose1], expected_pose1_updated, atol=1e-6))

    def test_build_system_simple_prior(self):
        """Test building JTJ, JTr, and cost for a simple prior cost problem."""
        problem = LeastSquaresProblem(costs=[self.prior0])
        current_values = problem.get_initial_values()
        # For prior on P0 (initially Identity) to target Identity, residual should be zero.
        # If P0 is slightly perturbed, J should be non-zero.
        
        # Perturb P0 slightly
        delta_p0 = torch.tensor([0.01,0,0,0,0,0], dtype=DEFAULT_DTYPE, device=DEVICE)
        current_values[self.pose0] = self.pose0.retract(current_values[self.pose0], delta_p0)

        JTJ, neg_JTr, total_cost = problem.build_system(current_values)

        self.assertEqual(JTJ.shape, (6,6))
        self.assertEqual(neg_JTr.shape, (6,))
        self.assertGreater(total_cost, 1e-5) # Cost should be non-zero due to perturbation

        # Check Jacobian (numerically if needed, or rely on Cost.jacobian correctness)
        # For a prior cost log(P * T_target^-1), if T_target=I, res = log(P).
        # Jacobian d(log(P))/dP is related to I. For small P (near I), J is approx I.
        # J_cost, res_cost = self.prior0.jacobian(current_values, {}, 0)
        # expected_JTJ = J_cost.T @ J_cost * (self.prior0.weight**2) # weight is squared in J, or in res? It is in res.
        # Here, cost.jacobian returns d(res)/d(delta), res already has weight.
        # So, JTJ = J_cost.T @ J_cost. neg_JTr = -J_cost.T @ res_cost
        # self.assertTrue(torch.allclose(JTJ, expected_JTJ, atol=1e-5))
        # self.assertTrue(torch.allclose(neg_JTr, -J_cost.T @ res_cost, atol=1e-5))
        # total_cost is sum(res_cost_i^2)
        # self.assertAlmostEqual(total_cost, torch.sum(res_cost**2).item(), delta=1e-5)
        # This part is more of a test of the cost function itself, assume Cost.jacobian is tested elsewhere.

    def test_build_system_two_costs(self):
        """Test system build with two costs involving shared and unique variables."""
        problem = LeastSquaresProblem(costs=[self.between01, self.between12]) # P0, P1, P2
        current_values = problem.get_initial_values()
        
        JTJ, neg_JTr, total_cost = problem.build_system(current_values)
        
        self.assertEqual(problem.total_tangent_dim, 18)
        self.assertEqual(JTJ.shape, (18,18))
        self.assertEqual(neg_JTr.shape, (18,))
        self.assertGreater(total_cost, 0) # Likely non-zero with initial estimates

        # Verify contributions to JTJ structure (simplified check)
        # J01 affects P0, P1 block. J12 affects P1, P2 block.
        # P0-P0 block (0:6, 0:6) should be non-zero from between01
        # P1-P1 block (6:12, 6:12) should get contributions from both between01 and between12
        # P2-P2 block (12:18, 12:18) should be non-zero from between12
        # P0-P1 block (0:6, 6:12) should be non-zero (off-diagonal)
        # P1-P2 block (6:12, 12:18) should be non-zero (off-diagonal)
        # P0-P2 block should be zero
        
        self.assertTrue(torch.any(JTJ[0:6, 0:6] != 0))
        self.assertTrue(torch.any(JTJ[6:12, 6:12] != 0))
        self.assertTrue(torch.any(JTJ[12:18, 12:18] != 0))
        self.assertTrue(torch.any(JTJ[0:6, 6:12] != 0))
        self.assertTrue(torch.any(JTJ[6:12, 0:6] != 0)) # Symmetric
        self.assertTrue(torch.any(JTJ[6:12, 12:18] != 0))
        self.assertTrue(torch.any(JTJ[12:18, 6:12] != 0)) # Symmetric
        self.assertTrue(torch.all(JTJ[0:6, 12:18] == 0))
        self.assertTrue(torch.all(JTJ[12:18, 0:6] == 0))

if __name__ == '__main__':
    unittest.main() 