import torch
import math
from typing import Dict, Optional

from .options import SolverOptions
from ..core.problem import LeastSquaresProblem
from ..variables.base import Variable
from ..utils.misc import DEVICE, DEFAULT_DTYPE

class GaussNewtonSolver:
    """
    Solves a least-squares problem using the Gauss-Newton algorithm.

    Args:
        problem (LeastSquaresProblem): The problem definition.
        options (Optional[SolverOptions], optional): Solver configuration. Defaults to SolverOptions().
    """
    def __init__(self, problem: LeastSquaresProblem, options: Optional[SolverOptions] = None):
        self.problem = problem
        self.options = options if options else SolverOptions()

    @torch.compile
    def solve(self) -> Dict[Variable, torch.Tensor]:
        """
        Executes the Gauss-Newton optimization algorithm.

        Returns:
            Dict[Variable, torch.Tensor]: A dictionary mapping optimized variables to their final tensor values.
        """
        current_values = self.problem.get_initial_values()
        
        if self.options.verbose:
            print("Starting Gauss-Newton Optimization")
            header = f"{'Iter':>4} | {'Cost':>12} | {'Cost Delta':>12} | {'Step Norm':>12} | {'Grad Norm':>12}"
            print(header)
            print("-" * len(header))
            
        prev_cost = float('inf')

        for i in range(self.options.max_iterations):
            # JTJ is (TangentDim, TangentDim), neg_JTr is (TangentDim,)
            JTJ, neg_JTr, current_cost = self.problem.build_system(current_values)

            if not math.isfinite(current_cost):
                if self.options.verbose: print(f"Error: Cost is {current_cost} at iteration {i}. Stopping.")
                break
            
            delta_tangent: torch.Tensor
            if self.problem.total_tangent_dim == 0: # No variables to optimize
                delta_tangent = torch.empty(0, device=DEVICE, dtype=DEFAULT_DTYPE)
            else:
                try:
                    # Add small identity matrix for regularization / numerical stability (damped Gauss-Newton)
                    # For pure Gauss-Newton, this damping factor (1e-8) should be ideally zero,
                    # but a small value helps with singular/ill-conditioned JTJ.
                    damping_factor = 1e-8 
                    I = torch.eye(JTJ.shape[0], device=JTJ.device, dtype=JTJ.dtype)
                    delta_tangent = torch.linalg.solve(JTJ + I * damping_factor, neg_JTr)
                except torch.linalg.LinAlgError:
                    if self.options.verbose: print(f"Warning: Singular matrix JTJ at iteration {i}. Using pseudo-inverse with damping.")
                    # Fallback to pseudo-inverse if solve fails
                    try:
                        damping_factor_pinv = 1e-7 # Slightly larger damping for pseudo-inverse
                        I_pinv = torch.eye(JTJ.shape[0], device=JTJ.device, dtype=JTJ.dtype)
                        delta_tangent = torch.linalg.pinv(JTJ + I_pinv * damping_factor_pinv) @ neg_JTr
                    except torch.linalg.LinAlgError:
                        if self.options.verbose: print(f"Error: Pseudo-inverse also failed at iteration {i}. Stopping.")
                        break # Stop if pseudo-inverse also fails
                except RuntimeError as e:
                     if self.options.verbose: print(f"Runtime error during linalg.solve: {e} at iter {i}. Stopping.")
                     break

            step_norm = torch.linalg.norm(delta_tangent).item() if delta_tangent.numel() > 0 else 0.0
            # neg_JTr is -J^T*r. The norm of the gradient of 0.5*||r||^2 is ||J^T*r||
            # So, grad_norm is norm(-neg_JTr) which is norm(JTr)
            grad_norm = torch.linalg.norm(neg_JTr).item() if neg_JTr.numel() > 0 else 0.0

            if not math.isfinite(grad_norm):
                if self.options.verbose: print(f"Error: Grad norm is {grad_norm} at iteration {i}. Stopping.")
                break

            cost_delta_abs = prev_cost - current_cost
            cost_delta_rel = cost_delta_abs / (abs(prev_cost) + 1e-12) if abs(prev_cost) > 1e-12 else cost_delta_abs

            if self.options.verbose:
                print(f"{i:4} | {current_cost:12.6e} | {cost_delta_abs:12.6e} | {step_norm:12.6e} | {grad_norm:12.6e}")
            
            # Convergence checks
            # Check 1: Cost reduction and step size are small
            cost_converged = (abs(cost_delta_abs) < self.options.tolerance_cost_delta_abs or \
                              abs(cost_delta_rel) < self.options.tolerance_cost_delta_rel)
            step_converged = step_norm < self.options.tolerance_step_norm
            
            if i > 0 and cost_converged and step_converged: # i > 0 because first iteration prev_cost is inf
                if self.options.verbose: print("Converged: Cost change and step norm below tolerance.")
                break
            
            # Check 2: Gradient norm is small
            if grad_norm < self.options.tolerance_grad_norm:
                 if self.options.verbose: print("Converged: Gradient norm below tolerance.")
                 break
            
            if delta_tangent.numel() > 0:
                current_values = self.problem.update_values(current_values, delta_tangent)
            elif self.problem.total_tangent_dim > 0 : # No step taken but there are variables
                if self.options.verbose: print(f"Warning: No step taken at iteration {i} despite non-zero tangent dim. Check problem conditioning.")
                # This might happen if JTJ is zero and neg_JTr is zero, implying local minimum or saddle point.
                break # Or continue if grad_norm is not yet met and hoping for a change later (unlikely for GN)

            prev_cost = current_cost
        else: # Loop finished without break (max_iterations reached)
            if self.options.verbose: print("Reached max iterations.")
            
        return current_values 