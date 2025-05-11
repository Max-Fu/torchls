import torch
import tyro # For CLI argument parsing
from typing import Literal

from torchls.variables.base import Variable # For reseting ID counter if needed for multiple runs
from torchls.variables.lie_groups import SE3Variable
from torchls.lie_math.se3 import se3_exp_map, se3_log_map
from torchls.core.problem import LeastSquaresProblem
from torchls.core.cost import Cost
from torchls.solvers.options import SolverOptions
from torchls.solvers.gauss_newton import GaussNewtonSolver
from torchls.solvers.lm import LevenbergMarquardtSolver
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# Define specific cost functions for the example problem (similar to test costs)
class ExampleSE3PriorCost(Cost):
    """Cost for a prior on an SE3 variable."""
    def __init__(self, variable: SE3Variable, target_pose: torch.Tensor, weight: float = 1.0):
        super().__init__([variable], name=f"Prior_{variable.name}")
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
        
        T_delta = torch.matmul(current_pose_b, torch.linalg.inv(target_pose_b))
        residual_log = se3_log_map(T_delta)
        return residual_log.flatten() * self.weight

class ExampleSE3BetweenCost(Cost):
    """Cost for a relative transformation constraint between two SE3 variables."""
    def __init__(self, var1: SE3Variable, var2: SE3Variable, measured_delta_T_12: torch.Tensor, weight: float = 1.0):
        super().__init__([var1, var2], name=f"Between_{var1.name}_{var2.name}")
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

def main(solver_type: Literal["gauss-newton", "lm"] = "lm", verbose: bool = True):
    """
    Runs a simple 2-pose SE(3) graph optimization problem.

    Args:
        solver_type: The type of solver to use ('gauss-newton' or 'lm').
        verbose: Whether to print solver progress and results.
    """
    print(f"TorchLS Simple Pose Graph Example - Using {DEVICE} with {DEFAULT_DTYPE}")
    print(f"Solver: {solver_type}")

    # Reset variable IDs for consistent naming if run multiple times in a session
    Variable._next_id = 0

    # 1. Create Variables
    pose0 = SE3Variable(name="World_Pose0") # Initialized to Identity
    
    # Initial estimate for pose1 is slightly off from its true measured relative pose from pose0
    initial_pose1_delta_from_world = torch.tensor([0.8, 0.1, -0.1, 0.05, -0.02, 0.03], dtype=DEFAULT_DTYPE)
    initial_pose1_T_world = se3_exp_map(initial_pose1_delta_from_world)
    pose1 = SE3Variable(initial_value=initial_pose1_T_world, name="World_Pose1")

    if verbose:
        print(f"Initial P0:\n{pose0.initial_value.squeeze().cpu().numpy()}")
        print(f"Initial P1:\n{pose1.initial_value.squeeze().cpu().numpy()}")

    # 2. Define Measurements and Costs
    # Prior on pose0 (anchor it to origin)
    prior_on_pose0 = ExampleSE3PriorCost(pose0, SE3Variable.identity(batch_size=()), weight=100.0)

    # Relative measurement between pose0 and pose1
    # True relative transformation T_0_1 (Pose0 to Pose1)
    true_T_0_1_delta = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=DEFAULT_DTYPE) # e.g. moved 1m in x
    measured_T_0_1 = se3_exp_map(true_T_0_1_delta)
    between_0_1 = ExampleSE3BetweenCost(pose0, pose1, measured_T_0_1, weight=1.0)

    # 3. Create Problem
    problem = LeastSquaresProblem(costs=[prior_on_pose0, between_0_1])

    # 4. Configure Solver
    options = SolverOptions(
        max_iterations=20,
        verbose=verbose,
        tolerance_grad_norm=1e-8,
        tolerance_step_norm=1e-8
    )

    # 5. Solve
    if solver_type == "gauss-newton":
        solver = GaussNewtonSolver(problem, options)
    elif solver_type == "lm":
        solver = LevenbergMarquardtSolver(problem, options)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    solution = solver.solve()

    # 6. Print Results
    if verbose:
        print("\n--- Optimization Results ---")
        final_pose0_val = solution[pose0]
        final_pose1_val = solution[pose1]
        print(f"Final P0:\n{final_pose0_val.squeeze().cpu().numpy()}")
        print(f"Final P1:\n{final_pose1_val.squeeze().cpu().numpy()}")

        # Verify results
        # Pose0 should be very close to Identity
        identity_check = torch.allclose(final_pose0_val.squeeze(), SE3Variable.identity(batch_size=()), atol=1e-5)
        print(f"P0 close to Identity: {identity_check}")

        # Relative transform P0_inv * P1 should be close to measured_T_0_1
        T_0_1_optimized = torch.linalg.inv(final_pose0_val) @ final_pose1_val
        relative_check = torch.allclose(T_0_1_optimized.squeeze(), measured_T_0_1, atol=1e-5)
        print(f"Optimized T_0_1:\n{T_0_1_optimized.squeeze().cpu().numpy()}")
        print(f"Measured T_0_1:\n{measured_T_0_1.squeeze().cpu().numpy()}")
        print(f"Relative P0->P1 close to measurement: {relative_check}")

if __name__ == "__main__":
    # Use tyro to parse CLI arguments for main function
    tyro.cli(main) 