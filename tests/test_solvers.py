import torch
import unittest
import io
import sys

from torchls.core.problem import LeastSquaresProblem
from torchls.core.cost import Cost # Not directly used, but good for context
from torchls.variables.base import Variable
from torchls.variables.lie_groups import SE3Variable
from torchls.lie_math.se3 import se3_exp_map, se3_log_map
from torchls.solvers.options import SolverOptions
from torchls.solvers.gauss_newton import GaussNewtonSolver
from torchls.solvers.lm import LevenbergMarquardtSolver
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# --- Helper Cost Classes (copied from test_core_problem.py for self-containment if needed, or import) ---
# To avoid import issues if tests are run individually, we can redefine them here
# Or, if these tests are always run together with test_core_problem, an import would be cleaner.
# For now, let's assume they might be run separately and redefine.

class SE3PriorCost(Cost):
    """Cost for a prior on an SE3 variable."""
    def __init__(self, variable: SE3Variable, target_pose: torch.Tensor, weight: float = 1.0):
        super().__init__([variable], name=f"TestPrior_{variable.name}")
        self.variable = variable
        if target_pose.ndim == 2: target_pose = target_pose.unsqueeze(0)
        self.target_pose = target_pose.to(device=DEVICE, dtype=DEFAULT_DTYPE)
        self.weight = weight

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        current_pose = var_values[self.variable]
        current_pose_b = current_pose if current_pose.ndim == 3 else current_pose.unsqueeze(0)
        target_pose_b = self.target_pose
        if target_pose_b.shape[0] == 1 and current_pose_b.shape[0] > 1:
            target_pose_b = target_pose_b.expand(current_pose_b.shape[0], -1, -1)
        # Using original formulation: T_delta = current_pose_b @ torch.linalg.inv(target_pose_b)
        T_delta = torch.matmul(current_pose_b, torch.linalg.inv(target_pose_b))
        residual_log = se3_log_map(T_delta)
        return residual_log.flatten() * self.weight

class SE3BetweenCost(Cost):
    """Cost for a relative transformation constraint between two SE3 variables."""
    def __init__(self, var1: SE3Variable, var2: SE3Variable, measured_delta_T_12: torch.Tensor, weight: float = 1.0):
        super().__init__([var1, var2], name=f"TestBetween_{var1.name}_{var2.name}")
        self.var1 = var1
        self.var2 = var2
        if measured_delta_T_12.ndim == 2: measured_delta_T_12 = measured_delta_T_12.unsqueeze(0)
        self.measured_delta_T_12 = measured_delta_T_12.to(device=DEVICE, dtype=DEFAULT_DTYPE)
        self.weight = weight

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        T_w1 = var_values[self.var1]
        T_w2 = var_values[self.var2]
        T_w1_b = T_w1 if T_w1.ndim == 3 else T_w1.unsqueeze(0)
        T_w2_b = T_w2 if T_w2.ndim == 3 else T_w2.unsqueeze(0)
        measured_delta_b = self.measured_delta_T_12
        if measured_delta_b.shape[0] == 1 and T_w1_b.shape[0] > 1:
            measured_delta_b = measured_delta_b.expand(T_w1_b.shape[0], -1, -1)
        if T_w1_b.shape[0] == 1 and T_w2_b.shape[0] > 1:
            T_w1_b = T_w1_b.expand(T_w2_b.shape[0],-1,-1)
        elif T_w2_b.shape[0] == 1 and T_w1_b.shape[0] > 1:
            T_w2_b = T_w2_b.expand(T_w1_b.shape[0],-1,-1)
        
        T_1_2_estimated = torch.matmul(torch.linalg.inv(T_w1_b), T_w2_b)
        error_transform = torch.matmul(T_1_2_estimated, torch.linalg.inv(measured_delta_b))
        residual_log = se3_log_map(error_transform)
        return residual_log.flatten() * self.weight

class TestSolvers(unittest.TestCase):
    """Tests for GaussNewtonSolver and LevenbergMarquardtSolver."""

    def _run_solver_test(self, solver_class, solver_name: str, verbose_tests: bool = False):
        """Helper function to run a simple pose graph optimization."""
        Variable._next_id = 3000 if solver_name == "GN" else 4000 # Avoid ID collision between tests
        
        pose0 = SE3Variable(name=f"P0_{solver_name}")
        # Slightly off initial guess for pose1
        init_pose1_delta_val = torch.tensor([0.1, -0.1, 0.1, 0.01, -0.01, 0.01], dtype=DEFAULT_DTYPE, device=DEVICE)
        init_pose1_val = se3_exp_map(init_pose1_delta_val)
        pose1 = SE3Variable(initial_value=init_pose1_val, name=f"P1_{solver_name}")

        # Prior on pose0 to be Identity
        prior_on_pose0 = SE3PriorCost(pose0, SE3Variable.identity(batch_size=()), weight=100.0)
        
        # Measurement for T_0_1 (pose0 to pose1)
        T_0_1_measured_true_delta = torch.tensor([1.0, 0.2, -0.1, 0.0, 0.05, -0.05], dtype=DEFAULT_DTYPE, device=DEVICE)
        T_0_1_measured = se3_exp_map(T_0_1_measured_true_delta) # (4,4)
        between_0_1 = SE3BetweenCost(pose0, pose1, T_0_1_measured, weight=1.0)

        problem = LeastSquaresProblem(costs=[prior_on_pose0, between_0_1])
        options = SolverOptions(verbose=verbose_tests, max_iterations=25,
                                tolerance_grad_norm=1e-7, tolerance_step_norm=1e-7,
                                tolerance_cost_delta_abs=1e-8, tolerance_cost_delta_rel=1e-8,
                                lambda_init=1e-4 if solver_name=="LM" else 1e-3) # LM might need smaller lambda_init
        
        solver = solver_class(problem, options)
        
        # Capture stdout if verbose_tests is True to check output formatting
        captured_output = io.StringIO()
        if verbose_tests:
            sys.stdout = captured_output
        
        solution = solver.solve()
        
        if verbose_tests:
            sys.stdout = sys.__stdout__ # Reset stdout
            output_str = captured_output.getvalue()
            # print(f"\n--- {solver_name} Verbose Output ---\n{output_str}") # Optional: print captured output
            self.assertIn("Iter", output_str) # Basic check for verbose header
            self.assertIn("Cost", output_str)
            if solver_name == "LM": self.assertIn("Lambda", output_str)

        final_pose0 = solution[pose0]
        final_pose1 = solution[pose1]

        # Check P0 is close to Identity
        self.assertTrue(torch.allclose(final_pose0.squeeze(0), SE3Variable.identity(batch_size=()), atol=1e-5),
                        f"{solver_name}: Final P0 not close to Identity.")

        # Check relative transform P0_inv * P1 is close to measured T_0_1
        T_final0_inv = torch.linalg.inv(final_pose0)
        T_final0_to_final1 = torch.matmul(T_final0_inv, final_pose1)
        self.assertTrue(torch.allclose(T_final0_to_final1.squeeze(0), T_0_1_measured, atol=1e-5),
                        f"{solver_name}: Final T_0_1 ({T_final0_to_final1.squeeze(0).cpu().numpy()}) not close to measured ({T_0_1_measured.cpu().numpy()}).")
        if verbose_tests: print(f"{solver_name} test passed with verbose output checks.")
        else: print(f"{solver_name} test passed.")

    def test_gauss_newton_solver(self):
        """Test the GaussNewtonSolver on a simple pose graph."""
        print("\n--- Testing Simple Pose Graph (Gauss-Newton) ---")
        self._run_solver_test(GaussNewtonSolver, "GN", verbose_tests=True)

    def test_levenberg_marquardt_solver(self):
        """Test the LevenbergMarquardtSolver on a simple pose graph."""
        print("\n--- Testing Simple Pose Graph (Levenberg-Marquardt) ---")
        self._run_solver_test(LevenbergMarquardtSolver, "LM", verbose_tests=True)
    
    def test_no_variables_problem(self):
        """Test solvers with a problem that has no optimizable (LieGroup) variables."""
        print("\n--- Testing Solvers with No Optimizable Variables ---")
        fixed_var = Variable(name="FixedOnly") # Not a LieGroupVariable
        # Dummy cost that uses this fixed var (residual can be anything, jacobian should be empty)
        class FixedCost(Cost):
            def __init__(self, var: Variable):
                super().__init__([var])
                self.var = var
            def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
                return torch.tensor([1.0], device=DEVICE, dtype=DEFAULT_DTYPE)
        
        problem = LeastSquaresProblem(costs=[FixedCost(fixed_var)])
        self.assertEqual(problem.total_tangent_dim, 0)

        # Test Gauss-Newton
        gn_solver = GaussNewtonSolver(problem, SolverOptions(verbose=False))
        sol_gn = gn_solver.solve()
        self.assertEqual(len(sol_gn), 0) # No Lie vars means empty solution dict for optimized vars
        print("GN with no optimizable vars: OK")

        # Test Levenberg-Marquardt
        lm_solver = LevenbergMarquardtSolver(problem, SolverOptions(verbose=False))
        sol_lm = lm_solver.solve()
        self.assertEqual(len(sol_lm), 0)
        print("LM with no optimizable vars: OK")

if __name__ == '__main__':
    print(f"TorchLS Tests - Using device: {DEVICE}, dtype: {DEFAULT_DTYPE}")
    unittest.main() 