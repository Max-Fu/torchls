import torch
from torchls.variables import SE3Variable
from torchls.core import Cost, LeastSquaresProblem
from torchls.solvers import GaussNewtonSolver, SolverOptions
from torchls.lie_math import se3_exp_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# 1. Define the SE(3) variable to be estimated, starting from identity
pose_to_estimate_gn = SE3Variable(name="EstimatedPoseGN")

# 2. Define a target SE(3) transformation (our "ground truth")
target_params = torch.tensor([0.1, -0.2, 0.15, 0.05, -0.03, 0.08], dtype=DEFAULT_DTYPE, device=DEVICE)
target_pose_matrix = se3_exp_map(target_params) # Shape (1,4,4)

# 3. Define a custom cost function
class PoseErrorCost(Cost):
    def __init__(self, estimated_var: SE3Variable, target_matrix: torch.Tensor):
        super().__init__(variables=[estimated_var], name="PoseErrorCost")
        self.target_matrix = target_matrix.to(DEVICE, DEFAULT_DTYPE)
        self._lie_var_helper = estimated_var # To call local_coordinates

    def residual(self, var_values: dict[SE3Variable, torch.Tensor]) -> torch.Tensor:
        current_est_matrix = var_values[self.variables[0]]
        # Residual = log(target @ current_estimate.inv())
        # This should be zero when current_estimate matches target_matrix.
        # local_coordinates(val1, val2) computes log(val2 @ val1.inv())
        res = self._lie_var_helper.local_coordinates(current_est_matrix, self.target_matrix)
        return res.squeeze(0) # Squeeze batch dim, result is (6,)

# 4. Create the cost instance and the problem
cost_gn = PoseErrorCost(pose_to_estimate_gn, target_pose_matrix)
problem_gn = LeastSquaresProblem(costs=[cost_gn])

# 5. Configure and run the Gauss-Newton solver
options = SolverOptions(verbose=False, max_iterations=10) # Set verbose=True to see steps
solver_gn = GaussNewtonSolver(problem_gn, options)
optimized_values_gn = solver_gn.solve()
final_pose_gn = optimized_values_gn[pose_to_estimate_gn]
print("Solution: ", final_pose_gn)

# 6. Verify (optional)
# print(f"Target Pose:\n{target_pose_matrix.squeeze()}\n")
# print(f"Estimated Pose (GN):\n{final_pose_gn.squeeze()}\n")
error_vector_gn = pose_to_estimate_gn.local_coordinates(final_pose_gn, target_pose_matrix)
# print(f"Final error norm (GN): {torch.linalg.norm(error_vector_gn).item():.2e}")
assert torch.allclose(final_pose_gn, target_pose_matrix, atol=1e-5)
print("Gauss-Newton example finished successfully.")