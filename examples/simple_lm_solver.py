import torch
from torchls.variables import SE3Variable
from torchls.core import Cost, LeastSquaresProblem
from torchls.solvers import LevenbergMarquardtSolver, SolverOptions # Changed solver
from torchls.lie_math import se3_exp_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# 1. Define the SE(3) variable to be estimated, starting from identity
pose_to_estimate_lm = SE3Variable(name="EstimatedPoseLM")

# 2. Define a target SE(3) transformation (same as GN example)
target_params = torch.tensor([0.1, -0.2, 0.15, 0.05, -0.03, 0.08], dtype=DEFAULT_DTYPE, device=DEVICE)
target_pose_matrix = se3_exp_map(target_params) # Shape (1,4,4)

# 3. Define a custom cost function (can reuse from GN example if defined globally)
class PoseErrorCost(Cost): # Redefined for self-containment, or ensure it's available
    def __init__(self, estimated_var: SE3Variable, target_matrix: torch.Tensor):
        super().__init__(variables=[estimated_var], name="PoseErrorCost")
        self.target_matrix = target_matrix.to(DEVICE, DEFAULT_DTYPE)
        self._lie_var_helper = estimated_var

    def residual(self, var_values: dict[SE3Variable, torch.Tensor]) -> torch.Tensor:
        current_est_matrix = var_values[self.variables[0]]
        res = self._lie_var_helper.local_coordinates(current_est_matrix, self.target_matrix)
        return res.squeeze(0)

# 4. Create the cost instance and the problem
cost_lm = PoseErrorCost(pose_to_estimate_lm, target_pose_matrix)
problem_lm = LeastSquaresProblem(costs=[cost_lm])

# 5. Configure and run the Levenberg-Marquardt solver
options = SolverOptions(verbose=False, max_iterations=10, lambda_init=1e-2) # Set verbose=True to see steps
solver_lm = LevenbergMarquardtSolver(problem_lm, options)
optimized_values_lm = solver_lm.solve()
final_pose_lm = optimized_values_lm[pose_to_estimate_lm]
print("Solution: ", final_pose_lm)

# 6. Verify (optional)
# print(f"Target Pose:\n{target_pose_matrix.squeeze()}\n")
# print(f"Estimated Pose (LM):\n{final_pose_lm.squeeze()}\n")
error_vector_lm = pose_to_estimate_lm.local_coordinates(final_pose_lm, target_pose_matrix)
# print(f"Final error norm (LM): {torch.linalg.norm(error_vector_lm).item():.2e}")
assert torch.allclose(final_pose_lm, target_pose_matrix, atol=1e-5)
print("Levenberg-Marquardt example finished successfully.")