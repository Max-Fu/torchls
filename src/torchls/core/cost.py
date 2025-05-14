import torch
import abc
from typing import List, Dict, Tuple, TYPE_CHECKING, Optional

from ..utils.misc import DEVICE, DEFAULT_DTYPE
# Ensure Variable and LieGroupVariable are available for type hints
from ..variables.base import Variable # Moved up for direct availability
from ..variables.lie_groups import LieGroupVariable # Moved up

# if TYPE_CHECKING: # Not strictly needed if imported directly above, but good practice
# from ..variables.base import Variable
# from ..variables.lie_groups import LieGroupVariable

class Cost(abc.ABC):
    """
    Abstract base class for cost functions in a least-squares problem.

    Args:
        variables (List[Variable]): A list of variables that this cost function depends on.
        name (str, optional): An optional name for the cost function.

    Attributes:
        variables (List[Variable]): The variables involved in this cost.
        name (str): Name of the cost function.
    """
    def __init__(self, variables: List[Variable], name: Optional[str] = None):
        self.variables = variables
        self.name = name if name else self.__class__.__name__
        self.dim: Optional[int] = None # Residual dimension, can be set by user or inferred

    @abc.abstractmethod
    def residual(self, var_values: Dict[Variable, torch.Tensor]) -> torch.Tensor:
        """
        Computes the residual vector for this cost function.

        Args:
            var_values (Dict[Variable, torch.Tensor]): A dictionary mapping variables
                to their current tensor values.

        Returns:
            torch.Tensor: The residual vector (e). Its shape is (ResidualDim,).
        """
        pass
    
    def analytical_jacobian(
        self, 
        var_values: Dict[Variable, torch.Tensor],
        ordered_lie_vars_for_cost: List[LieGroupVariable] # Added this argument
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Optionally implemented by subclasses to provide an analytical Jacobian.

        Args:
            var_values (Dict[Variable, torch.Tensor]): Current values of all variables in the problem.
            ordered_lie_vars_for_cost (List[LieGroupVariable]): Lie group variables relevant to this cost,
                                                                 in the order their Jacobian columns should appear.
                                                                 This is pre-filtered and sorted by problem-wide index.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]:
                - J_cost (torch.Tensor): The analytical Jacobian matrix for this cost (d_residual / d_delta_tangent).
                                         Shape (ResidualDim, CostTangentDim). CostTangentDim is sum of tangent_dims
                                         of `ordered_lie_vars_for_cost`.
                - residual (torch.Tensor): The residual vector evaluated at `var_values`.
                                           Shape (ResidualDim,).
            Return None or raise NotImplementedError if not implemented.
        """
        return None # Default implementation indicates not available

    @torch.compile
    def jacobian(self, 
                 var_values: Dict[Variable, torch.Tensor],
                 _variable_to_cols: Dict[Variable, Tuple[int, int]], 
                 _total_tangent_dim: int 
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Jacobian of the residual with respect to the tangent space
        of the Lie group variables involved in this cost, using automatic differentiation.
        The order of Jacobian blocks corresponds to the order of Lie group variables
        as they appear in `self.variables` after being filtered and sorted by their
        global problem-wide index (if available from `var_values.keys()`).

        Args:
            var_values (Dict[Variable, torch.Tensor]): Current values of all variables in the problem.
            _variable_to_cols (Dict[Variable, Tuple[int, int]]): Mapping from variable to its column
                                                               range in the full Jacobian (unused here).
            _total_tangent_dim (int): Total dimension of the tangent space for all variables (unused here).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - J_cost (torch.Tensor): The Jacobian matrix for this cost (d_residual / d_delta).
                                         Shape (ResidualDim, CostTangentDim).
                - residual (torch.Tensor): The residual vector evaluated at `var_values`.
                                           Shape (ResidualDim,).
        """
        # No need for lazy import here as they are imported at the top of the file
        
        problem_lie_vars_order_map = {var: i for i, var in enumerate(
            v for v in var_values.keys() if isinstance(v, LieGroupVariable)
        )}
        
        cost_lie_vars_in_problem_order: List[LieGroupVariable] = sorted(
            [v for v in self.variables if isinstance(v, LieGroupVariable) and v in problem_lie_vars_order_map],
            key=lambda v: problem_lie_vars_order_map[v]
        )

        # Attempt to use analytical Jacobian first
        analytical_result = self.analytical_jacobian(var_values, cost_lie_vars_in_problem_order)
        if analytical_result is not None:
            J_analytical, res_analytical = analytical_result
            # Basic validation (more can be added)
            if not isinstance(J_analytical, torch.Tensor) or not isinstance(res_analytical, torch.Tensor):
                raise ValueError("analytical_jacobian must return a tuple of two tensors.")
            
            # Infer residual dimension if not set from analytical result
            if self.dim is None and res_analytical.ndim > 0:
                 self.dim = res_analytical.shape[0]
            elif self.dim is None and res_analytical.ndim == 0: # scalar residual
                 self.dim = 1

            expected_cost_tangent_dim = sum(v.tangent_dim for v in cost_lie_vars_in_problem_order)
            
            if J_analytical.ndim != 2 or \
               (self.dim is not None and J_analytical.shape[0] != self.dim and self.dim > 0) or \
               (self.dim == 0 and J_analytical.shape[0] != 0) or \
               J_analytical.shape[1] != expected_cost_tangent_dim:
                # Allow (0,0) if dim is 0 and tangent_dim is 0
                if not (self.dim == 0 and expected_cost_tangent_dim == 0 and J_analytical.shape == (0,0)):
                     raise ValueError(
                        f"Analytical Jacobian for cost '{self.name}' has incorrect shape. "
                        f"Expected ({self.dim if self.dim is not None else 'auto'}, {expected_cost_tangent_dim}), "
                        f"Got {J_analytical.shape}. Residual shape: {res_analytical.shape}."
                    )
            if self.dim is not None and res_analytical.shape[0] != self.dim and self.dim > 0:
                if not (self.dim == 0 and res_analytical.numel() == 0): # allow (0,) or empty for dim 0
                    raise ValueError(
                        f"Analytical residual for cost '{self.name}' has incorrect shape. "
                        f"Expected ({self.dim if self.dim is not None else 'auto'},), Got {res_analytical.shape}."
                    )
            return J_analytical, res_analytical

        # Fallback to autograd if analytical_jacobian is not implemented or returns None
        if not cost_lie_vars_in_problem_order:
            res = self.residual(var_values)
            # Infer res_dim if Cost.dim is not set
            if self.dim is None:
                self.dim = res.shape[0] if res.ndim > 0 and res.numel() > 0 else 0
            
            # Use self.dim for consistency
            current_res_dim = self.dim if self.dim is not None else 0
            
            return torch.empty((current_res_dim, 0), device=DEVICE, dtype=DEFAULT_DTYPE), res
        
        def residual_wrt_delta_closure(*deltas_for_vars_in_cost: torch.Tensor) -> torch.Tensor: 
            temp_var_values_for_residual = var_values.copy()
            for i, var_lie_lgv in enumerate(cost_lie_vars_in_problem_order):
                current_val = var_values[var_lie_lgv]
                delta_val = deltas_for_vars_in_cost[i]
                temp_var_values_for_residual[var_lie_lgv] = var_lie_lgv.retract(current_val, delta_val)
            return self.residual(temp_var_values_for_residual)
            
        zero_deltas_for_autograd_list = []
        for var_lie_lgv_outer in cost_lie_vars_in_problem_order:
            delta_v = torch.zeros(var_lie_lgv_outer.tangent_dim, device=DEVICE, dtype=DEFAULT_DTYPE, requires_grad=True)
            zero_deltas_for_autograd_list.append(delta_v)
        
        if not zero_deltas_for_autograd_list: # Should not happen if cost_lie_vars_in_problem_order is not empty
            res_val = self.residual(var_values)
            current_res_dim = res_val.shape[0] if res_val.ndim > 0 and res_val.numel() > 0 else 0
            if self.dim is None: self.dim = current_res_dim
            return torch.zeros((self.dim if self.dim is not None else 0, 0), device=DEVICE, dtype=DEFAULT_DTYPE), res_val
        
        zero_deltas_for_autograd_tuple = tuple(zero_deltas_for_autograd_list)
        
        jac_blocks_tuple = torch.autograd.functional.jacobian(residual_wrt_delta_closure, 
                                                               zero_deltas_for_autograd_tuple, 
                                                               strict=True, vectorize=False, create_graph=False)
        
        if not isinstance(jac_blocks_tuple, tuple): 
            jac_blocks_tuple = (jac_blocks_tuple,)
        
        valid_jacobian_blocks = []
        # Get current residual value to determine its dimension
        current_res_val = self.residual(var_values) # This is the residual at the linearization point
        
        # Infer and set self.dim if not already set (e.g. by analytical_jacobian or user)
        if self.dim is None:
            self.dim = current_res_val.shape[0] if current_res_val.ndim > 0 and current_res_val.numel() > 0 else 0
        
        # Use self.dim for consistent shaping
        res_dim_to_use = self.dim if self.dim is not None else 0

        for i, jac_block_i in enumerate(jac_blocks_tuple):
            var_lie_i = cost_lie_vars_in_problem_order[i]
            expected_tangent_dim_i = var_lie_i.tangent_dim
        
            if jac_block_i is None: # Autograd might return None if a variable does not affect residual
                jac_block_i = torch.zeros((res_dim_to_use, expected_tangent_dim_i), device=DEVICE, dtype=DEFAULT_DTYPE)
            
            # Reshape jac_block_i if it's 1D due to scalar residual or scalar tangent space
            if res_dim_to_use == 1 and jac_block_i.ndim == 1 and jac_block_i.shape[0] == expected_tangent_dim_i:
                jac_block_i = jac_block_i.unsqueeze(0)
            elif jac_block_i.ndim == 1 and expected_tangent_dim_i == 1 and jac_block_i.shape[0] == res_dim_to_use:
                jac_block_i = jac_block_i.unsqueeze(1)
            elif jac_block_i.ndim == 0 and expected_tangent_dim_i == 0 and res_dim_to_use == 0:
                 jac_block_i = torch.empty((0,0), device=DEVICE, dtype=DEFAULT_DTYPE)
            
            if jac_block_i.shape != (res_dim_to_use, expected_tangent_dim_i):
                # Attempt to reshape if dimensions match, otherwise raise error
                try:
                    # Handle cases like (0,) or (,0) if res_dim or tangent_dim is 0
                    if res_dim_to_use == 0 and expected_tangent_dim_i == 0 and jac_block_i.numel() == 0:
                         jac_block_i = torch.empty((0,0), device=DEVICE, dtype=DEFAULT_DTYPE)
                    elif res_dim_to_use > 0 and expected_tangent_dim_i == 0 and jac_block_i.shape == (res_dim_to_use,): # e.g. (N,) for (N,0)
                         jac_block_i = torch.empty((res_dim_to_use,0), device=DEVICE, dtype=DEFAULT_DTYPE)
                    elif res_dim_to_use == 0 and expected_tangent_dim_i > 0 and jac_block_i.shape == (expected_tangent_dim_i,): # e.g. (M,) for (0,M)
                         jac_block_i = torch.empty((0,expected_tangent_dim_i), device=DEVICE, dtype=DEFAULT_DTYPE)
                    else:
                        jac_block_i = jac_block_i.reshape(res_dim_to_use, expected_tangent_dim_i)
                except RuntimeError as e:
                    raise ValueError(
                        f"Jacobian block for {var_lie_i.name} has unexpected shape {jac_block_i.shape}. "
                        f"Expected ({res_dim_to_use}, {expected_tangent_dim_i}). "
                        f"Autograd output: {jac_blocks_tuple[i].shape if i < len(jac_blocks_tuple) and jac_blocks_tuple[i] is not None else 'N/A'}. Error: {e}"
                    )
        
            valid_jacobian_blocks.append(jac_block_i)
        
        if not valid_jacobian_blocks: 
             return torch.zeros((res_dim_to_use, 0), device=DEVICE, dtype=DEFAULT_DTYPE), current_res_val
        
        J_for_cost = torch.cat(valid_jacobian_blocks, dim=1)
        return J_for_cost, current_res_val 