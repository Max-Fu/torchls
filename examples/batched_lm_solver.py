import torch
from torchls.variables import SE3Variable # Ensure Variable is imported for type hints if used
from torchls.core import Cost, LeastSquaresProblem
from torchls.solvers import LevenbergMarquardtSolver, SolverOptions
from torchls.lie_math import se3_exp_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# Define the PoseErrorCost class (as used in previous examples)
class PoseErrorCost(Cost):
    def __init__(self, estimated_var: SE3Variable, target_matrix: torch.Tensor):
        # Using a unique name for each cost based on the variable it's associated with
        super().__init__(variables=[estimated_var], name=f"PoseErrorCost_{estimated_var.name}")
        self.target_matrix = target_matrix.to(DEVICE, DEFAULT_DTYPE) # Target is (1,4,4)
        self._lie_var_helper = estimated_var # To call local_coordinates method

    def residual(self, var_values: dict[SE3Variable, torch.Tensor]) -> torch.Tensor:
        current_est_matrix = var_values[self.variables[0]] # Current estimate (1,4,4)
        # local_coordinates for SE3Variable with (1,4,4) inputs returns (1,6)
        res = self._lie_var_helper.local_coordinates(current_est_matrix, self.target_matrix)
        return res.squeeze(0) # Return shape (6,) for this single pose error

# 1. Define Batch Size
BATCH_SIZE = 2

# 2. Create a list of SE3Variable instances (one for each item in the batch)
# Each starts at identity and will be optimized independently.
poses_to_estimate_batch = [
    SE3Variable(name=f"BatchPose_{i}") for i in range(BATCH_SIZE)
]

# 3. Create a list of target SE(3) matrices (one for each variable)
# Each target_param corresponds to one pose in the batch.
target_params_list = [
    torch.tensor([ 0.1, -0.2,  0.15, 0.05, -0.03,  0.08], dtype=DEFAULT_DTYPE, device=DEVICE),
    torch.tensor([-0.05, 0.1, -0.1, 0.02,  0.06, -0.04], dtype=DEFAULT_DTYPE, device=DEVICE)
]
assert len(target_params_list) == BATCH_SIZE
# target_pose_matrices_list will be a list of (1,4,4) tensors
target_pose_matrices_list = [se3_exp_map(params) for params in target_params_list]

# 4. Create a list of Cost functions
# Each cost links one SE3Variable to its corresponding target matrix.
all_costs_for_batch = []
for i in range(BATCH_SIZE):
    cost_i = PoseErrorCost(poses_to_estimate_batch[i], target_pose_matrices_list[i])
    all_costs_for_batch.append(cost_i)

# 5. Create the LeastSquaresProblem
# This problem now includes BATCH_SIZE * 6 degrees of freedom.
problem_batch = LeastSquaresProblem(costs=all_costs_for_batch)

# 6. Configure and run the Levenberg-Marquardt solver
options_batch = SolverOptions(verbose=False, max_iterations=10, lambda_init=1e-2)
solver_batch = LevenbergMarquardtSolver(problem_batch, options_batch)
optimized_values_batch = solver_batch.solve()

# 7. Verify each pose in the batch
print(f"Batched LM example (B={BATCH_SIZE}) finished successfully.")
for i in range(BATCH_SIZE):
    estimated_pose_var = poses_to_estimate_batch[i]
    final_pose_tensor = optimized_values_batch[estimated_pose_var]
    print(f"Found solution for pose {i}: ", final_pose_tensor)
    target_pose_tensor = target_pose_matrices_list[i]
    
    # Optional: Print target and final estimate for each pose
    # print(f"\nTarget Pose {i}:\n{target_pose_tensor.squeeze()}\n")
    # print(f"Estimated Pose {i} (LM):\n{final_pose_tensor.squeeze()}\n")
    
    error_vector = estimated_pose_var.local_coordinates(final_pose_tensor, target_pose_tensor)
    # print(f"Final error norm for pose {i} (LM): {torch.linalg.norm(error_vector).item():.2e}")
    assert torch.allclose(final_pose_tensor, target_pose_tensor, atol=1e-5)

print("All poses in the batch converged to their targets.")