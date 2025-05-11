# torchls

A PyTorch library for non-linear least squares optimization, with a focus on problems involving Lie groups. It reimplements the core functionality of [jaxls](https://github.com/brentyi/jaxls) in PyTorch.

## Installation

To install `torchls` locally, clone this repository and install it using pip:

```bash
git clone https://github.com/Max-Fu/torchls
cd torchls
pip install -e .
```

## Usage Examples

Here are a few examples of how to use the `SE3Variable` from the `torchls` package.

### 1. Creating an SE3Variable

You can create an `SE3Variable` representing a 3D rigid body transformation.

```python
import torch
from torchls.variables import SE3Variable
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE # For device and dtype consistency

# Create an SE3Variable initialized to identity
# Internally, it's stored as a (1, 4, 4) tensor
se3_identity = SE3Variable(name="Pose_Identity")
print("Identity SE3 Variable:\n", se3_identity.initial_value)

# Create an SE3Variable from a given 4x4 matrix
# The input can be a (4,4) tensor
T_matrix = torch.tensor([
    [0.0, -1.0, 0.0, 1.0],
    [1.0,  0.0, 0.0, 2.0],
    [0.0,  0.0, 1.0, 3.0],
    [0.0,  0.0, 0.0, 1.0]
], device=DEVICE, dtype=DEFAULT_DTYPE)
se3_from_matrix = SE3Variable(initial_value=T_matrix, name="Pose_Matrix")
print("\nSE3 Variable from matrix:\n", se3_from_matrix.initial_value)

# Create a batch of SE3Variables
T_batch = torch.stack([torch.eye(4, device=DEVICE, dtype=DEFAULT_DTYPE)] * 3) # (3,4,4)
se3_batch = SE3Variable(initial_value=T_batch, name="Pose_Batch")
print("\nBatched SE3 Variable:\n", se3_batch.initial_value)
```

### 2. Retracting an SE3Variable (Updating a Pose)

The `retract` method updates the SE(3) transformation using a tangent space vector (6D for SE(3)). This is often used in optimization to apply an update.
`T_new = Exp(delta) * T_current`

```python
import torch
from torchls.variables import SE3Variable
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# Initial SE(3) variable (identity)
current_pose = SE3Variable(name="CurrentPose") # (1,4,4)
T_current = current_pose.initial_value

# Delta (update in tangent space - e.g., small translation and rotation)
# [vx, vy, vz, wx, wy, wz]
delta_tangent = torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], device=DEVICE, dtype=DEFAULT_DTYPE) # (6,)

# Perform retraction
T_updated = current_pose.retract(T_current, delta_tangent)
print("Original Pose (Identity):\n", T_current)
print("\nTangent Update (delta):\n", delta_tangent)
print("\nUpdated Pose (T_new):\n", T_updated)
assert T_updated.shape == (1,4,4) # Retaining batch dim
```

### 3. Calculating Local Coordinates (Difference Between Poses)

The `local_coordinates` method computes the tangent space vector that transforms `value1` to `value2`.
`delta = Log(value2 @ value1.inv)`

```python
import torch
from torchls.variables import SE3Variable
from torchls.lie_math import se3_exp_map
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE

# Two SE(3) poses
T1 = SE3Variable(name="Pose1").initial_value # Identity (1,4,4)

delta_for_T2 = torch.tensor([0.1, 0.2, 0.3, 0.04, 0.05, 0.06], device=DEVICE, dtype=DEFAULT_DTYPE)
T2_mat = se3_exp_map(delta_for_T2) # (1,4,4) matrix
T2 = SE3Variable(initial_value=T2_mat, name="Pose2").initial_value

# Calculate local coordinates (tangent vector from T1 to T2)
# We need an instance of SE3Variable to call its method
helper_var = SE3Variable()
delta_recovered = helper_var.local_coordinates(T1, T2)

print("Pose T1:\n", T1)
print("\nPose T2 (T1 @ Exp(delta_for_T2)):\n", T2)
print("\nOriginal delta_for_T2:\n", delta_for_T2)
print("\nRecovered delta (local_coordinates(T1, T2)):\n", delta_recovered)
# Check if the recovered delta is close to the original one
assert torch.allclose(delta_recovered.squeeze(0), delta_for_T2, atol=1e-6)
```

### 4. Example: Gauss-Newton Solver

This example demonstrates how to use the `GaussNewtonSolver` to estimate an SE(3) pose that matches a target pose.

```python
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

# 6. Verify (optional)
# print(f"Target Pose:\n{target_pose_matrix.squeeze()}\n")
# print(f"Estimated Pose (GN):\n{final_pose_gn.squeeze()}\n")
error_vector_gn = pose_to_estimate_gn.local_coordinates(final_pose_gn, target_pose_matrix)
# print(f"Final error norm (GN): {torch.linalg.norm(error_vector_gn).item():.2e}")
assert torch.allclose(final_pose_gn, target_pose_matrix, atol=1e-5)
print("Gauss-Newton example finished successfully.")
```

### 5. Example: Levenberg-Marquardt Solver

This example uses the `LevenbergMarquardtSolver` for the same pose estimation problem.

```python
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

# 6. Verify (optional)
# print(f"Target Pose:\n{target_pose_matrix.squeeze()}\n")
# print(f"Estimated Pose (LM):\n{final_pose_lm.squeeze()}\n")
error_vector_lm = pose_to_estimate_lm.local_coordinates(final_pose_lm, target_pose_matrix)
# print(f"Final error norm (LM): {torch.linalg.norm(error_vector_lm).item():.2e}")
assert torch.allclose(final_pose_lm, target_pose_matrix, atol=1e-5)
print("Levenberg-Marquardt example finished successfully.")
```

### 6. Example: Batched Optimization (Solving for Multiple Poses with Levenberg-Marquardt)

This example shows how to optimize a batch of SE(3) poses simultaneously. Each pose in the batch will be estimated against its own target. We achieve this by defining multiple `SE3Variable` instances and corresponding `Cost` functions, then solving them together in one `LeastSquaresProblem`.

```python
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
```

## Acknowledgements

This library is a torch reimplementation of [jaxls](https://github.com/brentyi/jaxls) by goated [Brent Yi](https://github.com/brentyi). Cursor and Gemini 2.5 Pro helped me a lot.