import argparse
import time
import numpy as np
import torch
# Enable torch.compiler allow_in_graph for sparse_coo_tensor
torch.compiler.allow_in_graph(torch.sparse_coo_tensor)
import jax
# Enable JAX x64 mode for float64 precision consistency
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_dataclasses as jdc
from typing import Optional, Union, Tuple, Dict, List # Added for RnTorchVariable

# torchls imports
import torchls.core.cost as torchls_cost
import torchls.core.problem as torchls_problem
import torchls.variables.lie_groups as torchls_lie_groups # Import the module
from torchls.solvers.options import SolverOptions
from torchls.variables import Variable
import torchls.solvers.gauss_newton as torchls_gn
import torchls.solvers.lm as torchls_lm
from torchls.utils.misc import DEVICE, DEFAULT_DTYPE # Added for RnTorchVariable
from torchls.sparse import SparseCsrMatrix, SparseCooMatrix  # Import sparse matrix implementations

# jaxls imports
import jaxls
from jaxls import LeastSquaresProblem, TerminationConfig, VarValues # These are top-level
from jaxls._solvers import NonlinearSolver # Import from submodule

# --- Custom optimized LeastSquaresProblem with better sparse matrix handling ---
class OptimizedLeastSquaresProblem(torchls_problem.LeastSquaresProblem):
    """
    Optimized version of LeastSquaresProblem that uses sparse matrix operations
    for better performance when building the linear system.
    """
    
    @torch.compile
    def build_system(self, current_values: Dict[Variable, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Constructs the Gauss-Newton system (JTJ * dx = -JTr) and calculates the total cost
        using efficient sparse matrix operations.
        
        Args:
            current_values (Dict[Variable, torch.Tensor]): Current values of all variables.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]:
                - JTJ (torch.Tensor): The J^T * J matrix. Shape (TotalTangentDim, TotalTangentDim).
                - neg_JTr (torch.Tensor): The -J^T * r vector. Shape (TotalTangentDim,).
                - total_cost (float): The sum of squared residuals.
        """
        if not self._is_analyzed: self._analyze()
        
        # Early return for zero tangent dimension
        if self.total_tangent_dim == 0:
            return (torch.empty((0,0), device=DEVICE, dtype=DEFAULT_DTYPE),
                    torch.empty(0, device=DEVICE, dtype=DEFAULT_DTYPE),
                    0.0)
        
        # Pre-calculate all residuals to get total number of residuals
        residuals_all_costs_list = []
        num_residuals_total = 0
        total_cost = 0.0
        
        for cost_item in self.costs:
            cost_var_values = {var_k: current_values[var_k] 
                              for var_k in cost_item.variables 
                              if var_k in current_values}
            
            residual = cost_item.residual(cost_var_values)
            if residual.ndim == 0:
                residual = residual.unsqueeze(0)
                
            residuals_all_costs_list.append(residual)
            num_residuals_total += residual.shape[0]
            total_cost += torch.sum(residual**2).item()
        
        # For very small problems, dense computations might be faster
        if num_residuals_total < 100 and self.total_tangent_dim < 10:
            return self._build_system_dense(current_values, residuals_all_costs_list, total_cost)
        else:
            return self._build_system_sparse(current_values, residuals_all_costs_list, total_cost)
    
    def _build_system_dense(self, current_values, residuals_all_costs_list, total_cost):
        """Dense implementation of the system building"""
        J_full = torch.zeros((sum(r.shape[0] for r in residuals_all_costs_list), 
                            self.total_tangent_dim), device=DEVICE, dtype=DEFAULT_DTYPE)
        r_full = torch.cat(residuals_all_costs_list)
        
        current_row_offset = 0
        for cost_idx, cost_item in enumerate(self.costs):
            cost_var_values = {var_k: current_values[var_k] 
                              for var_k in cost_item.variables 
                              if var_k in current_values}
            
            J_cost, _ = cost_item.jacobian(cost_var_values, {}, 0)
            num_residuals_this_cost = residuals_all_costs_list[cost_idx].shape[0]
            
            # Get Lie vars in this cost, in problem order
            vars_in_cost = sorted(
                [v for v in cost_item.variables if isinstance(v, torchls_lie_groups.LieGroupVariable) 
                 and v in self.var_to_tangent_info],
                key=lambda v_lie: self.var_to_tangent_info[v_lie][0]
            )
            
            current_col_offset = 0
            for var_lie in vars_in_cost:
                _, global_start_col, var_tangent_dim = self.var_to_tangent_info[var_lie]
                
                if var_tangent_dim > 0 and num_residuals_this_cost > 0 and J_cost.numel() > 0:
                    jac_slice = J_cost[:, current_col_offset:current_col_offset + var_tangent_dim]
                    J_full[current_row_offset:current_row_offset + num_residuals_this_cost,
                           global_start_col:global_start_col + var_tangent_dim] = jac_slice
                    
                current_col_offset += var_tangent_dim
            current_row_offset += num_residuals_this_cost
        
        # Compute JTJ and -JTr
        JTJ = J_full.T @ J_full
        neg_JTr = -J_full.T @ r_full
        
        return JTJ, neg_JTr, total_cost
    
    def _build_system_sparse(self, current_values, residuals_all_costs_list, total_cost):
        """Sparse implementation of the system building using CSR format"""
        # Lists to collect sparse matrix data
        row_indices = []
        col_indices = []
        values = []
        
        r_full = torch.cat(residuals_all_costs_list)
        
        current_row_offset = 0
        for cost_idx, cost_item in enumerate(self.costs):
            cost_var_values = {var_k: current_values[var_k] 
                              for var_k in cost_item.variables 
                              if var_k in current_values}
            
            J_cost, _ = cost_item.jacobian(cost_var_values, {}, 0)
            num_residuals_this_cost = residuals_all_costs_list[cost_idx].shape[0]
            
            # Get Lie vars in this cost, in problem order
            vars_in_cost = sorted(
                [v for v in cost_item.variables if isinstance(v, torchls_lie_groups.LieGroupVariable) 
                 and v in self.var_to_tangent_info],
                key=lambda v_lie: self.var_to_tangent_info[v_lie][0]
            )
            
            current_col_offset = 0
            for var_lie in vars_in_cost:
                _, global_start_col, var_tangent_dim = self.var_to_tangent_info[var_lie]
                
                if var_tangent_dim > 0 and num_residuals_this_cost > 0 and J_cost.numel() > 0:
                    jac_slice = J_cost[:, current_col_offset:current_col_offset + var_tangent_dim]
                    
                    # Find non-zero elements in the slice
                    non_zero_mask = jac_slice != 0
                    if torch.any(non_zero_mask):
                        non_zero_rows, non_zero_cols = torch.where(non_zero_mask)
                        
                        # Adjust indices for the full matrix
                        adjusted_rows = non_zero_rows + current_row_offset
                        adjusted_cols = non_zero_cols + global_start_col
                        
                        # Collect values and indices
                        row_indices.append(adjusted_rows)
                        col_indices.append(adjusted_cols)
                        values.append(jac_slice[non_zero_mask])
                    
                current_col_offset += var_tangent_dim
            current_row_offset += num_residuals_this_cost
        
        # If no non-zero elements, return empty matrices
        if not values:
            JTJ = torch.zeros((self.total_tangent_dim, self.total_tangent_dim), 
                             device=DEVICE, dtype=DEFAULT_DTYPE)
            neg_JTr = torch.zeros(self.total_tangent_dim, device=DEVICE, dtype=DEFAULT_DTYPE)
            return JTJ, neg_JTr, total_cost
        
        # Create sparse COO matrix for J
        all_values = torch.cat(values)
        all_row_indices = torch.cat(row_indices)
        all_col_indices = torch.cat(col_indices)
        
        # Create sparse Jacobian
        J_sparse = SparseCooMatrix(
            values=all_values,
            row_indices=all_row_indices,
            col_indices=all_col_indices,
            shape=(r_full.shape[0], self.total_tangent_dim)
        )
        
        # Convert to PyTorch sparse format
        J_torch_sparse = J_sparse.to_torch_sparse_coo()
        
        # Compute J^T * J using sparse operations
        JTJ = torch.sparse.mm(J_torch_sparse.transpose(0, 1), J_torch_sparse).to_dense()
        
        # Compute -J^T * r = -(J^T * r)
        neg_JTr = -torch.sparse.mm(J_torch_sparse.transpose(0, 1), r_full.unsqueeze(1)).squeeze(1)
        
        return JTJ, neg_JTr, total_cost

# --- Define RnTorchVariable for torchls ---
class RnTorchVariable(torchls_lie_groups.LieGroupVariable):
    # n_dim: int, dimension of the R^n space
    def __init__(self, initial_value: torch.Tensor, name: str = ""):
        # initial_value: torch.Tensor of shape (n_dim,) or (1, n_dim)
        # n_dim is inferred from initial_value.shape[-1]
        if initial_value.ndim == 1:
            initial_value = initial_value.unsqueeze(0) # Store as (1, n_dim)
        super().__init__(initial_value, name)
        self._tangent_dim = initial_value.shape[-1]

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim

    def retract(self, current_value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # current_value: (B, D) or (D,)
        # delta: (B, D) or (D,)
        return current_value + delta

    def local_coordinates(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        # value1: (B, D) or (D,)
        # value2: (B, D) or (D,)
        return value2 - value1

    @classmethod
    def identity(cls, n_dim: int, batch_size: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
        # n_dim: dimension of the R^n space
        # batch_size behavior:
        #   None or 1 -> (1, n_dim)
        #   0 or () -> (n_dim,)
        #   int > 1 -> (batch_size, n_dim)
        #   tuple -> (*batch_size, n_dim)
        id_vec = torch.zeros(n_dim, device=DEVICE, dtype=DEFAULT_DTYPE)
        if batch_size is None or (isinstance(batch_size, int) and batch_size == 1):
            return id_vec.unsqueeze(0)
        if (isinstance(batch_size, int) and batch_size == 0) or \
           (isinstance(batch_size, tuple) and not batch_size):
            return id_vec
        if isinstance(batch_size, int): # batch_size > 1
            return id_vec.unsqueeze(0).expand(batch_size, -1)
        if isinstance(batch_size, tuple): # tuple batch_size
             return id_vec.unsqueeze(0).expand(*batch_size, -1)
        return id_vec # Should not happen with current logic

# --- Define JAX Variables ---
class JaxParamM(jaxls.Var, default_factory=lambda: jnp.array(0.0, dtype=jnp.float64)): # Scalar default
    pass # tangent_dim will be inferred as 1, retract is Euclidean

class JaxParamC(jaxls.Var, default_factory=lambda: jnp.array(0.0, dtype=jnp.float64)): # Scalar default
    pass # tangent_dim will be inferred as 1, retract is Euclidean

# JAX cost factory
@jaxls.Cost.create_factory
def compute_regression_residual(var_values: VarValues, 
                                A_matrix: jnp.ndarray, 
                                y_vector: jnp.ndarray, 
                                m_var_ref: JaxParamM, 
                                c_var_ref: JaxParamC) -> jnp.ndarray:
    # var_values: Current values from solver
    # A_matrix, y_vector: Static data
    # m_var_ref, c_var_ref: Template Var instances for keying
    m_val = var_values[m_var_ref]  # This is a JAX scalar (shape ())
    c_val = var_values[c_var_ref]  # This is a JAX scalar (shape ())
    
    # A_matrix is (problem_size,), y_vector is (problem_size,)
    # m_val and c_val are scalar.
    # Broadcasting A_matrix * m_val will work as expected.
    return A_matrix * m_val + c_val - y_vector

# Helper to define torchls problem
class TorchLinearRegressionCost(torchls_cost.Cost):
    # m_var: RnTorchVariable for slope
    # c_var: RnTorchVariable for intercept
    # x_val: float, single x data point
    # y_val: float, single y data point
    def __init__(self, m_var: RnTorchVariable, c_var: RnTorchVariable,
                 x_val: float, y_val: float):
        super().__init__([m_var, c_var], name=f"cost_pt_{x_val}")
        var_device = m_var.initial_value.device
        self.x_val_tensor = torch.tensor([x_val], device=var_device, dtype=torch.float64)
        self.y_val_tensor = torch.tensor([y_val], device=var_device, dtype=torch.float64)
        self.m_var = m_var
        self.c_var = c_var

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        m_val = var_values[self.m_var] # Shape (1,1) or (1,)
        c_val = var_values[self.c_var] # Shape (1,1) or (1,)
        # Ensure scalar multiplication if m_val/c_val are (1,1) and x_val_tensor is (1,)
        return (m_val.squeeze(-1) * self.x_val_tensor + c_val.squeeze(-1)) - self.y_val_tensor

# New vectorized cost implementation that handles all data points at once
class VectorizedTorchLinearRegressionCost(torchls_cost.Cost):
    """
    Vectorized implementation of linear regression cost that handles all data points at once.
    This is more efficient than creating a separate cost for each data point.
    """
    def __init__(self, m_var: RnTorchVariable, c_var: RnTorchVariable,
                 x_data: np.ndarray, y_data: np.ndarray):
        super().__init__([m_var, c_var], name="vectorized_linear_regression_cost")
        var_device = m_var.initial_value.device
        # Convert data to tensors
        self.x_data_tensor = torch.tensor(x_data, device=var_device, dtype=torch.float64)
        self.y_data_tensor = torch.tensor(y_data, device=var_device, dtype=torch.float64)
        self.m_var = m_var
        self.c_var = c_var

    def residual(self, var_values: dict[Variable, torch.Tensor]) -> torch.Tensor:
        m_val = var_values[self.m_var] # Shape (1,1) or (1,)
        c_val = var_values[self.c_var] # Shape (1,1) or (1,)
        
        # Vectorized computation for all data points at once
        # m_val and c_val are scalars, x_data_tensor is (N,), y_data_tensor is (N,)
        # Broadcasting will handle the computation across all points
        return (m_val.squeeze(-1) * self.x_data_tensor + c_val.squeeze(-1)) - self.y_data_tensor
    
    def analytical_jacobian(
        self, 
        var_values: Dict[Variable, torch.Tensor],
        ordered_lie_vars_for_cost: List[torchls_lie_groups.LieGroupVariable]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Provide an analytical Jacobian for the linear regression cost.
        
        For linear regression y = m*x + c, the Jacobian with respect to m and c is:
        - ∂r/∂m = x
        - ∂r/∂c = 1
        
        Where r is the residual vector (m*x + c - y).
        
        Args:
            var_values: Dictionary mapping variables to tensor values
            ordered_lie_vars_for_cost: Ordered list of Lie group variables relevant to this cost
            
        Returns:
            Tuple of (Jacobian matrix, residual vector)
        """
        # Calculate residual
        residual = self.residual(var_values)
        
        # Get number of data points
        n_points = self.x_data_tensor.shape[0]
        
        # Initialize Jacobian matrix with zeros
        total_tangent_dim = sum(var.tangent_dim for var in ordered_lie_vars_for_cost)
        J = torch.zeros((n_points, total_tangent_dim), device=self.x_data_tensor.device, dtype=self.x_data_tensor.dtype)
        
        # Fill Jacobian matrix for each variable
        col_idx = 0
        for var in ordered_lie_vars_for_cost:
            if var is self.m_var:
                # For m, derivative is x
                J[:, col_idx:col_idx + 1] = self.x_data_tensor.unsqueeze(1)
                col_idx += var.tangent_dim
            elif var is self.c_var:
                # For c, derivative is 1
                J[:, col_idx:col_idx + 1] = torch.ones_like(self.x_data_tensor).unsqueeze(1)
                col_idx += var.tangent_dim
                
        return J, residual


def setup_torchls_problem(x_data: np.ndarray, y_data: np.ndarray,
                          initial_m: float, initial_c: float,
                          device: torch.device, solver_type: str,
                          max_solver_iter: int):
    m_var = RnTorchVariable(torch.tensor([initial_m], device=device, dtype=torch.float64), name="m")
    c_var = RnTorchVariable(torch.tensor([initial_c], device=device, dtype=torch.float64), name="c")

    costs = []
    for i in range(len(x_data)):
        costs.append(TorchLinearRegressionCost(m_var, c_var, x_data[i], y_data[i]))
    
    problem = torchls_problem.LeastSquaresProblem(costs)
    
    solver_options = SolverOptions(max_iterations=max_solver_iter, verbose=False) # Use provided max_iter

    if solver_type == "gn":
        solver = torchls_gn.GaussNewtonSolver(problem, options=solver_options)
    elif solver_type == "lm":
        solver = torchls_lm.LevenbergMarquardtSolver(problem, options=solver_options)
    else:
        raise ValueError(f"Unknown torchls solver type: {solver_type}")

    initial_values = problem.get_initial_values() 
    return solver, initial_values

# New setup function that uses the vectorized cost
def setup_torchls_problem_vectorized(x_data: np.ndarray, y_data: np.ndarray,
                                     initial_m: float, initial_c: float,
                                     device: torch.device, solver_type: str,
                                     max_solver_iter: int,
                                     use_optimized_problem: bool = False):
    m_var = RnTorchVariable(torch.tensor([initial_m], device=device, dtype=torch.float64), name="m")
    c_var = RnTorchVariable(torch.tensor([initial_c], device=device, dtype=torch.float64), name="c")

    # Create a single vectorized cost instead of one per data point
    vectorized_cost = VectorizedTorchLinearRegressionCost(m_var, c_var, x_data, y_data)
    
    # Use either standard or optimized problem class
    if use_optimized_problem:
        problem = OptimizedLeastSquaresProblem([vectorized_cost])
    else:
        problem = torchls_problem.LeastSquaresProblem([vectorized_cost])
    
    solver_options = SolverOptions(max_iterations=max_solver_iter, verbose=False)

    if solver_type == "gn":
        solver = torchls_gn.GaussNewtonSolver(problem, options=solver_options)
    elif solver_type == "lm":
        solver = torchls_lm.LevenbergMarquardtSolver(problem, options=solver_options)
    else:
        raise ValueError(f"Unknown torchls solver type: {solver_type}")

    initial_values = problem.get_initial_values() 
    return solver, initial_values

# Helper to define jaxls problem
@jdc.pytree_dataclass
class JaxStaticParams:
    x_data: jax.Array
    y_data: jax.Array

# Modified setup_jaxls_problem
def setup_jaxls_problem(
    data_jax: Tuple[jnp.ndarray, jnp.ndarray],
    true_params_jax: Tuple[jnp.ndarray, jnp.ndarray], # true_params_jax not strictly needed for setup if initial_guess is fixed
    num_iters: int, # num_iters will be used in benchmark_jaxls for TerminationConfig
    problem_size: int,
    # solver_type: str = "gauss_newton", # No longer needed here, solver options passed to analyzed_problem.solve()
):
    # """Sets up the JAXLS problem for linear regression using the recommended API."""
    A_jax, y_jax = data_jax

    # JAX variables (scalar definitions)
    m_var = JaxParamM(jnp.array(0.0, dtype=jnp.float64)) 
    c_var = JaxParamC(jnp.array(0.0, dtype=jnp.float64)) 

    # Initial guess for the variables
    # VarValues.make can take Var instances; it will use their default_factory if not VarWithValue
    # Or, to be explicit with the initial value passed to JaxParamM/C:
    initial_guess_jax = VarValues.make([
        m_var.with_value(jnp.array(0.0, dtype=jnp.float64)), # Explicit initial value
        c_var.with_value(jnp.array(0.0, dtype=jnp.float64))  # Explicit initial value
    ])
    # Alternatively, if JaxParamM/C's positional arg truly sets the initial internal value for .make():
    # initial_guess_jax = VarValues.make([m_var, c_var])


    # Create cost instance using the factory
    # The factory captures A_jax, y_jax, m_var, c_var as part of the Cost object's 'args'
    cost_instance = compute_regression_residual(A_jax, y_jax, m_var, c_var)
    
    # Define the problem
    problem = LeastSquaresProblem(costs=[cost_instance], variables=[m_var, c_var])
    
    # Analyze the problem
    analyzed_problem = problem.analyze()

    # No explicit solver instantiation here; will use analyzed_problem.solve()
    # true_params_jax and data_jax are returned for consistency if benchmark_jaxls needs them for other purposes
    return analyzed_problem, initial_guess_jax, true_params_jax, data_jax


def benchmark_solver(solver_name: str, solver_obj, initial_vals, 
                     num_iterations_solve: int,
                     num_warmup_runs: int, num_benchmark_runs: int, 
                     device_name: str, is_jax: bool = False):
    
    print(f"\n--- Benchmarking {solver_name} on {device_name} ---")
    print(f"Solver configured for max {num_iterations_solve} iterations per solve call.")

    for _ in range(num_warmup_runs):
        if is_jax:
            # For JAX, solver.solve now takes initial_values directly
            # The problem is already part of the solver instance
            result_warmup = solver_obj.solve(initial_vals)
            jax.block_until_ready(result_warmup)
        else:
            _ = solver_obj.solve() # torchls solve takes no args if values already set
    
    solve_times = []
    for i in range(num_benchmark_runs):
        start_time = time.perf_counter()
        if is_jax:
            # JAX solver.solve takes initial_values
            solution = solver_obj.solve(initial_vals)
            jax.block_until_ready(solution)
        else:
            solution = solver_obj.solve()
            if isinstance(initial_vals, dict) and initial_vals and \
               next(iter(initial_vals.values())).device.type == 'cuda':
                torch.cuda.synchronize()
            elif torch.is_tensor(initial_vals) and initial_vals.device.type == 'cuda':
                 torch.cuda.synchronize()
        end_time = time.perf_counter()
        run_time = end_time - start_time
        solve_times.append(run_time)

    avg_time = sum(solve_times) / num_benchmark_runs
    min_time = min(solve_times)
    max_time = max(solve_times)
    
    print(f"Finished {num_benchmark_runs} benchmark runs.")
    print(f"Avg solve time: {avg_time:.6f} seconds")
    print(f"Min solve time: {min_time:.6f} seconds")
    print(f"Max solve time: {max_time:.6f} seconds")
    return avg_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark torchls vs jaxls for linear regression.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device to run on (cpu or cuda). CUDA must be available for PyTorch and JAX.")
    parser.add_argument("--num_points", type=int, default=100, help="Number of data points for regression.")
    parser.add_argument("--solver_iterations", type=int, default=10, 
                        help="Number of iterations the solver itself runs per solve() call.")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of warm-up runs before timing.")
    parser.add_argument("--benchmark_runs", type=int, default=20, help="Number of timed benchmark runs for averaging.")
    parser.add_argument("--solver_type", type=str, default="gn", choices=["gn", "lm"], help="Solver type for torchls (gn or lm). jaxls uses its NonlinearSolver.")
    parser.add_argument("--use_vectorized", action="store_true", help="Use vectorized implementation for torchls")
    parser.add_argument("--use_optimized", action="store_true", help="Use optimized sparse matrix operations")

    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available for PyTorch. Exiting.")
            return
        try:
            _ = jax.devices("gpu")
        except RuntimeError:
            print("CUDA not available for JAX (or JAX not built with GPU support). Exiting.")
            return
        torch_device = torch.device("cuda")
        print(f"JAX backend: {jax.default_backend()}")
        if jax.default_backend() != 'gpu':
             print("Warning: JAX is not using GPU backend despite --device cuda. Check JAX installation.")
    else:
        torch_device = torch.device("cpu")
        jax.config.update('jax_platform_name', 'cpu')
        print(f"JAX backend: {jax.default_backend()}")

    print(f"Benchmarking on: PyTorch device = {torch_device}, JAX using = {jax.default_backend()}")
    print(f"Problem: Linear Regression with {args.num_points} points.")
    print(f"Solver iterations per call: {args.solver_iterations}")
    print(f"Warmup runs: {args.warmup_runs}, Benchmark runs: {args.benchmark_runs}")
    print(f"Using vectorized implementation: {args.use_vectorized}")
    print(f"Using optimized sparse operations: {args.use_optimized}")

    np.random.seed(0)
    m_true, c_true = 2.5, 1.0
    x_data_np = np.random.rand(args.num_points) * 10
    y_data_np = m_true * x_data_np + c_true + np.random.randn(args.num_points) * 0.5
    initial_m, initial_c = 0.0, 0.0

    try:
        if args.use_vectorized:
            torchls_solver, torchls_initial_values = setup_torchls_problem_vectorized(
                x_data_np, y_data_np, initial_m, initial_c, 
                torch_device, args.solver_type, args.solver_iterations,
                use_optimized_problem=args.use_optimized
            )
            if args.use_optimized:
                solver_name = f"torchls ({args.solver_type}, vectorized, optimized)"
            else:
                solver_name = f"torchls ({args.solver_type}, vectorized)"
        else:
            torchls_solver, torchls_initial_values = setup_torchls_problem(
                x_data_np, y_data_np, initial_m, initial_c, 
                torch_device, args.solver_type, args.solver_iterations
            )
            solver_name = f"torchls ({args.solver_type})"
            
        benchmark_solver(solver_name, torchls_solver, torchls_initial_values, 
                        args.solver_iterations, args.warmup_runs, args.benchmark_runs, 
                        str(torch_device), is_jax=False)
    except Exception as e:
        print(f"Error during torchls benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        jaxls_analyzed_problem, jaxls_initial_guess, true_params_jax, data_jax = setup_jaxls_problem(
            (jnp.array(x_data_np, dtype=jnp.float64), jnp.array(y_data_np, dtype=jnp.float64)),
            (jnp.array([m_true], dtype=jnp.float64), jnp.array([c_true], dtype=jnp.float64)),
            args.solver_iterations, args.num_points
        )
        benchmark_solver("jaxls", jaxls_analyzed_problem, jaxls_initial_guess, 
                         args.solver_iterations, args.warmup_runs, args.benchmark_runs, 
                         jax.default_backend(), is_jax=True)
    except Exception as e:
        print(f"Error during jaxls benchmark: {e}")
        import traceback
        traceback.print_exc()

def benchmark_jaxls(
    data_jax: Tuple[jnp.ndarray, jnp.ndarray],
    true_params_jax: Tuple[jnp.ndarray, jnp.ndarray],
    num_iters: int,
    num_runs: int,
    problem_size: int,
):
    # """Benchmarks JAXLS solver using analyzed_problem.solve()."""
    analyzed_problem, initial_guess_jax, _, _ = setup_jaxls_problem( # Unpack only what's needed
        data_jax, true_params_jax, num_iters, problem_size
    )

    # Define termination configuration
    # Adding a small tolerance as it's good practice, can be adjusted or removed
    term_cfg = TerminationConfig(max_iterations=num_iters, parameter_tolerance=1e-6) 

    # Warm-up run
    _ = analyzed_problem.solve(
        initial_vals=initial_guess_jax,
        termination=term_cfg,
        linear_solver="cholmod", # Example, adjust as needed
        sparse_mode="blockrow",  # Example, adjust as needed
        verbose=False,
        return_summary=False
    )
    jax.block_until_ready(_)

    durations = []
    for _ in range(num_runs):
        start_time = time.time()
        solution = analyzed_problem.solve(
            initial_vals=initial_guess_jax,
            termination=term_cfg,
            linear_solver="cholmod", # Example, adjust as needed
            sparse_mode="blockrow",  # Example, adjust as needed
            verbose=False,
            return_summary=False
        )
        jax.block_until_ready(solution) # Ensure computation is finished
        durations.append(time.time() - start_time)
    return sum(durations) / num_runs

if __name__ == "__main__":
    main() 