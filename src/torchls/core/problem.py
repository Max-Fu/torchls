import torch
import math
from typing import List, Dict, Tuple, OrderedDict as OrderedDictType # Python 3.7+
from collections import OrderedDict

from .cost import Cost
from ..variables.base import Variable
from ..variables.lie_groups import LieGroupVariable
from ..utils.misc import DEVICE, DEFAULT_DTYPE

class LeastSquaresProblem:
    """
    Defines a least-squares optimization problem.
    It consists of a set of cost functions and the variables they depend on.

    Args:
        costs (List[Cost]): A list of Cost objects that define the problem.

    Attributes:
        costs (List[Cost]): The cost functions forming the problem.
        variables (OrderedDictType[Variable, None]): An ordered dictionary of unique
            Lie group variables involved in the optimization, maintaining insertion order.
        var_to_tangent_info (Dict[Variable, Tuple[int, int, int]]): Maps each Lie group
            variable to a tuple containing its index in the ordered list, its starting
            column in the Jacobian, and its tangent dimension.
        ordered_lie_vars_for_problem (List[LieGroupVariable]): All unique Lie group variables
            involved in the problem, in a fixed order.
        total_tangent_dim (int): The total dimension of the tangent space for all
            Lie group variables being optimized.
    """
    def __init__(self, costs: List[Cost]):
        self.costs = costs
        self.variables: OrderedDictType[Variable, None] = OrderedDict()
        self.var_to_tangent_info: Dict[Variable, Tuple[int, int, int]] = {}
        self.ordered_lie_vars_for_problem: List[LieGroupVariable] = []
        self.total_tangent_dim: int = 0
        self._is_analyzed = False
        self._analyze()

    def _analyze(self):
        """
        Analyzes the problem structure to identify all unique Lie group variables
        and determine their order and tangent space dimensions.
        This populates `ordered_lie_vars_for_problem`, `variables` (OrderedDict),
        `var_to_tangent_info`, and `total_tangent_dim`.
        """
        if self._is_analyzed: return
        
        # Collect all unique LieGroupVariables from all costs, maintaining an order
        # The order is based on first appearance across all costs, then by variable ID if needed for tie-breaking (implicitly handled by set properties initially)
        # Using a list and set to ensure uniqueness while trying to preserve some order from costs
        temp_seen_lie_vars_set = set()
        temp_ordered_lie_vars_list = []
        for cost_item in self.costs:
            for var_item in cost_item.variables:
                if isinstance(var_item, LieGroupVariable) and var_item not in temp_seen_lie_vars_set:
                    temp_ordered_lie_vars_list.append(var_item)
                    temp_seen_lie_vars_set.add(var_item)
        
        # Sort them by ID to ensure a canonical order if initial order was ambiguous or not desired
        # self.ordered_lie_vars_for_problem = sorted(list(temp_seen_lie_vars), key=lambda v: v.id)
        self.ordered_lie_vars_for_problem = temp_ordered_lie_vars_list # Maintain order of first appearance
        
        # Populate the OrderedDict for self.variables using the now fixed order
        self.variables = OrderedDict((var, None) for var in self.ordered_lie_vars_for_problem)

        current_tangent_dim_offset = 0
        for idx, var_lie in enumerate(self.ordered_lie_vars_for_problem):
            tangent_dim = var_lie.tangent_dim
            self.var_to_tangent_info[var_lie] = (idx, current_tangent_dim_offset, tangent_dim)
            current_tangent_dim_offset += tangent_dim
        
        self.total_tangent_dim = current_tangent_dim_offset
        self._is_analyzed = True

    def get_initial_values(self) -> Dict[Variable, torch.Tensor]:
        """
        Retrieves the initial values for all variables involved in the problem.
        For Lie group variables, it's their `initial_value` attribute.
        For other (e.g., fixed) variables, it attempts to get `initial_value` if present.

        Returns:
            Dict[Variable, torch.Tensor]: A dictionary mapping each variable to its initial tensor value.
        """
        if not self._is_analyzed: self._analyze()
        values = {}
        for var_lie in self.ordered_lie_vars_for_problem: 
            values[var_lie] = var_lie.initial_value.clone().detach()
        
        # Also collect initial values for non-Lie variables if they are part of any cost
        # and have an initial_value attribute (e.g. for fixed parameters)
        all_vars_in_costs = set(v for c in self.costs for v in c.variables)
        for var_c in all_vars_in_costs:
            if var_c not in values and hasattr(var_c, 'initial_value') and var_c.initial_value is not None:
                # Ensure it's a tensor and on the correct device/dtype if it's a torch tensor
                val = var_c.initial_value
                if isinstance(val, torch.Tensor):
                    values[var_c] = val.to(device=DEVICE, dtype=DEFAULT_DTYPE).clone().detach()
                else: # Store other types as is (e.g. if it was a Python float/int for a fixed param)
                    values[var_c] = val 
            elif var_c not in values: # Variable doesn't have initial_value, store None or raise error?
                 # For now, we only include variables with initial values.
                 # If a cost needs a variable not in `values`, it might error during residual/Jacobian. 
                 pass 
        return values

    def update_values(self, current_values: Dict[Variable, torch.Tensor], delta_tangent: torch.Tensor) -> Dict[Variable, torch.Tensor]:
        """
        Updates the values of Lie group variables given a tangent space update vector.

        Args:
            current_values (Dict[Variable, torch.Tensor]): The current values of all variables.
            delta_tangent (torch.Tensor): The tangent space update vector for all Lie group variables,
                                          concatenated in their established order. Shape (TotalTangentDim,).

        Returns:
            Dict[Variable, torch.Tensor]: A new dictionary with updated variable values.
        """
        if not self._is_analyzed: self._analyze()
        new_values = current_values.copy()
        for var_lie in self.ordered_lie_vars_for_problem: 
            _var_idx, start_col, tangent_dim = self.var_to_tangent_info[var_lie]
            if tangent_dim > 0: # Only update if there's a corresponding delta part
                # Ensure delta_tangent is not empty and slice is valid
                if delta_tangent.numel() > 0 and start_col + tangent_dim <= delta_tangent.shape[0]:
                    delta_var = delta_tangent[start_col : start_col + tangent_dim]
                    new_values[var_lie] = var_lie.retract(current_values[var_lie], delta_var)
                elif tangent_dim > 0: # delta_tangent is empty or too small, but var expects update
                    # This case might indicate an issue if delta_tangent is unexpectedly empty.
                    # For now, we assume if tangent_dim > 0, a valid delta_var should exist.
                    # If delta_tangent is empty, this var won't be updated, which might be fine if total_tangent_dim is 0.
                    pass 
        return new_values

    def build_system(self, current_values: Dict[Variable, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Constructs the Gauss-Newton system (JTJ * dx = -JTr) and calculates the total cost.
        J is the Jacobian of all residuals wrt all tangent space variables.
        r is the vector of all residuals.

        Args:
            current_values (Dict[Variable, torch.Tensor]): Current values of all variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]:
                - JTJ (torch.Tensor): The J^T * J matrix. Shape (TotalTangentDim, TotalTangentDim).
                - neg_JTr (torch.Tensor): The -J^T * r vector. Shape (TotalTangentDim,).
                - total_cost (float): The sum of squared residuals (0.5 * r^T * r is often used, but here just r^T * r).
        """
        if not self._is_analyzed: self._analyze()
        
        # Pre-calculate all residuals to get total number of residuals
        residuals_all_costs_list: List[torch.Tensor] = []
        num_residuals_total = 0
        for cost_item in self.costs:
            # Ensure all variables required by the cost are present in current_values
            cost_var_values_for_res = {var_k: current_values[var_k] 
                                       for var_k in cost_item.variables 
                                       if var_k in current_values}
            # Check if any variable was missing for this cost
            if len(cost_var_values_for_res) != len(cost_item.variables):
                missing_vars = [v.name for v in cost_item.variables if v not in cost_var_values_for_res]
                # This is a stricter check. The original code allowed jacobian to proceed.
                # Consider how to handle this: error, or let jacobian handle it (might get wrong dim J_cost)
                # For now, matching original behavior of passing available vars.
                # raise ValueError(f"Cost '{cost_item.name}' is missing variable(s) {missing_vars} in current_values for residual calculation.")
                pass # Continue, residual might still work or error if it accesses a missing key directly.

            r_cost_i = cost_item.residual(cost_var_values_for_res)
            if r_cost_i.ndim == 0: r_cost_i = r_cost_i.unsqueeze(0) # Ensure 1D for cat
            residuals_all_costs_list.append(r_cost_i)
            num_residuals_total += r_cost_i.shape[0]

        J_full = torch.zeros((num_residuals_total, self.total_tangent_dim), device=DEVICE, dtype=DEFAULT_DTYPE)
        # Ensure r_full is 1D even if num_residuals_total is 0
        r_full = torch.zeros(max(num_residuals_total, 0), device=DEVICE, dtype=DEFAULT_DTYPE)
        
        current_row_offset = 0
        current_total_sum_sq_residuals = 0.0 # This is sum(e_i^2), not 0.5 * sum(e_i^2)

        for cost_idx, cost_item in enumerate(self.costs):
            # Pass only the relevant subset of current_values to the jacobian method of the cost
            cost_var_values_for_jac = {var_k: current_values[var_k] 
                                       for var_k in cost_item.variables 
                                       if var_k in current_values}
            
            # The jacobian method of Cost now handles its own variable filtering and ordering internally
            J_cost, r_cost_at_lin_point = cost_item.jacobian(cost_var_values_for_jac, {}, 0) 
            # r_cost_at_lin_point is the residual computed at current_values (same as in residuals_all_costs_list)
            # Use the pre-computed residual for consistency and to fill r_full
            r_cost_actual = residuals_all_costs_list[cost_idx]
            num_residuals_this_cost = r_cost_actual.shape[0]

            if num_residuals_this_cost > 0 :
                r_full[current_row_offset : current_row_offset + num_residuals_this_cost] = r_cost_actual.flatten()
            current_total_sum_sq_residuals += torch.sum(r_cost_actual**2).item()
            
            # Assemble J_cost into J_full
            # The Cost.jacobian method returns J_cost where columns correspond to Lie vars in that cost,
            # ordered by their problem-wide index (which we established in _analyze and Cost.jacobian uses).
            # We need to place these blocks into J_full according to global_start_col.

            # Get Lie vars in this cost, in problem order (same as Cost.jacobian would have used for its output columns)
            # This is needed to map J_cost columns to J_full columns.
            vars_in_cost_that_are_lie_and_in_problem = sorted(
                [v for v in cost_item.variables if isinstance(v, LieGroupVariable) and v in self.var_to_tangent_info],
                key=lambda v_lie: self.var_to_tangent_info[v_lie][0] # Sort by problem-wide index
            )

            current_col_offset_within_J_cost = 0
            for var_lie_in_cost in vars_in_cost_that_are_lie_and_in_problem:
                _problem_var_idx, global_start_col, var_tangent_dim = self.var_to_tangent_info[var_lie_in_cost]
                
                if var_tangent_dim > 0 and num_residuals_this_cost > 0:
                    # Ensure J_cost has enough columns. J_cost.shape[1] should be sum of tangent_dims of vars_in_cost_that_are_lie_and_in_problem
                    if J_cost.numel() > 0 and J_cost.shape[1] >= current_col_offset_within_J_cost + var_tangent_dim:
                        jac_slice_for_var = J_cost[:, current_col_offset_within_J_cost : current_col_offset_within_J_cost + var_tangent_dim]
                        J_full[current_row_offset : current_row_offset + num_residuals_this_cost,
                               global_start_col : global_start_col + var_tangent_dim] = jac_slice_for_var
                    elif J_cost.numel() == 0 and J_cost.shape[1] == 0 and var_tangent_dim > 0: # J_cost is (res_dim, 0) but should have cols
                        # This implies an issue in Cost.jacobian logic if it returned an empty J for non-empty Lie vars.
                        # For now, this means no Jacobian contribution for this var from this cost.
                        pass 
                current_col_offset_within_J_cost += var_tangent_dim
            current_row_offset += num_residuals_this_cost
            
        # System: JTJ * dx = -JTr (or JTr for some formulations, here -JTr because we want to minimize cost)
        # Cost function is E = sum(r_i^2). Gradient is 2 * J^T * r. Hessian is approx 2 * J^T * J.
        # So, JTJ * dx = -JTr. The provided code uses JTr for the RHS.
        # The original code returned neg_JTr = J_full.T @ r_full, then later neg_JTr again.
        # This implies the solver expects JTr. If solver expects -JTr, then RHS is -JTr.
        # The provided code calculates `neg_JTr` as `J_full.T @ r_full` and then immediately returns `-neg_JTr`.
        # Let's stick to `JTr = J_full.T @ r_full` and return that, making the solver use `delta = solve(JTJ, JTr)`
        # The original script's GaussNewton used `delta_tangent = torch.linalg.solve(JTJ + I * 1e-8, neg_JTr)`
        # where `neg_JTr` was passed from `build_system`. And `build_system` returned `J_full.T @ r_full` as its `neg_JTr` item.
        # This implies that the solver expects the RHS of `JTJ dx = RHS` to be `J^T r`.
        # The optimization step is `x_{k+1} = x_k - (J^T J)^{-1} J^T r_k` which means `(J^T J) dx = -J^T r_k` if `dx = x_{k+1} - x_k`.
        # However, the local parameterization means `x_{k+1} = x_k [+] delta_x`, where `delta_x` is from tangent space.
        # The linear system is `(J^T J) delta_x = -J^T r`.
        # So, the RHS should be `-J^T r`.
        # The original script returned `JTJ, J_full.T @ r_full, current_total_cost` (where second term was named `neg_JTr` by mistake in naming, but value was `JTr`).
        # And the solver used it as `delta_tangent = torch.linalg.solve(JTJ + I * 1e-8, neg_JTr)`.
        # This means the `neg_JTr` variable in the solver was actually `JTr`.
        # To make `delta_tangent` be the correct step, the RHS for `solve` should be `-JTr`.
        # So, `build_system` should return `-J^T r`.

        if J_full.numel() == 0: # Handles case where total_tangent_dim is 0
            JTJ = torch.empty((0,0), device=DEVICE, dtype=DEFAULT_DTYPE)
            final_JTr = torch.empty(0, device=DEVICE, dtype=DEFAULT_DTYPE)
        else:
            JTJ = J_full.T @ J_full
            final_JTr = J_full.T @ r_full # This is J^T * r

        # The solver (GaussNewton) expects the RHS as is, and it solves JTJ * dx = RHS.
        # If RHS is -JTr, then dx is the update. The original code seems to use JTr as the RHS.
        # Let's follow the original code's variable name `neg_JTr` but assign `-J_full.T @ r_full` to it.
        # This will make `delta_tangent = torch.linalg.solve(JTJ, neg_JTr)` correct for `delta_x`.
        # The original script returned `J_full.T @ r_full` as `neg_JTr` to the solver,
        # then the solver used `delta = solve(JTJ, neg_JTr)`. This `delta` is for `x_new = x_old + delta` (additive sense)
        # or `X_new = X_old * Exp(delta)` (Lie group sense via retract, if delta is `-(J^TJ)^-1 J^T r`).
        # The standard GN update is `delta = -(H^-1)g`. Here `g = J^T r` (if cost is 0.5 r^T r, then `g = J^T r`).
        # If cost is `r^T r`, then `g = 2 J^T r`. Hessian `H = J^T J` (approx, if cost is 0.5 r^T r).
        # Or `H = 2 J^T J` if cost is `r^T r`.
        # System `H dx = -g`. If `H=JTJ`, `g=JTr`, then `JTJ dx = -JTr`.
        # The original script seems to have effectively solved `JTJ dx = JTr` by passing `JTr` as `neg_JTr`.
        # Let's provide `-JTr` as the second component, consistent with the standard formula for `dx`.

        return JTJ, -final_JTr, current_total_sum_sq_residuals 