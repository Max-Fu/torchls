import torch
import math
from typing import Dict, Optional

from .options import SolverOptions
from ..core.problem import LeastSquaresProblem
from ..variables.base import Variable
from ..utils.misc import DEVICE, DEFAULT_DTYPE

class LevenbergMarquardtSolver:
    """
    Solves a least-squares problem using the Levenberg-Marquardt algorithm.

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
        Executes the Levenberg-Marquardt optimization algorithm.

        Returns:
            Dict[Variable, torch.Tensor]: A dictionary mapping optimized variables to their final tensor values.
        """
        current_values = self.problem.get_initial_values()
        current_lambda = self.options.lambda_init

        if self.options.verbose:
            print("Starting Levenberg-Marquardt Optimization")
            header = f"{'Iter':>4} | {'Cost':>12} | {'Lambda':>10} | {'Step Qual.':>10} | {'Step Norm':>12} | {'Grad Norm':>12}"
            print(header)
            print("-" * len(header))
        
        # Initial system build
        # JTJ is (TangentDim, TangentDim), neg_JTr is (TangentDim,)
        JTJ, neg_JTr, current_cost = self.problem.build_system(current_values)

        if not math.isfinite(current_cost):
            if self.options.verbose: print(f"Initial cost is {current_cost}. Stopping.")
            return current_values
        
        # grad_norm = ||-neg_JTr|| = ||JTr||
        current_grad_norm = torch.linalg.norm(neg_JTr).item() if neg_JTr.numel() > 0 else 0.0
        if current_grad_norm < self.options.tolerance_grad_norm:
            if self.options.verbose: print("Converged: Initial gradient norm below tolerance.")
            return current_values

        for i in range(self.options.max_iterations):
            eye_P: torch.Tensor
            if self.problem.total_tangent_dim == 0:
                # No variables to optimize, system matrices will be empty
                eye_P = torch.empty((0,0), device=DEVICE, dtype=DEFAULT_DTYPE)
                delta_tangent = torch.empty(0, device=DEVICE, dtype=DEFAULT_DTYPE)
            else:
                eye_P = torch.eye(JTJ.shape[0], device=DEVICE, dtype=DEFAULT_DTYPE)
                try:
                    # System: (JTJ + lambda * I) * delta = neg_JTr (which is -J^T r)
                    delta_tangent = torch.linalg.solve(JTJ + current_lambda * eye_P, neg_JTr)
                except torch.linalg.LinAlgError:
                    if self.options.verbose: print(f"Warning: Singular matrix (JTJ + lambda*I) at iter {i} with lambda={current_lambda:.2e}. Increasing lambda.")
                    current_lambda = min(current_lambda * self.options.lambda_factor, self.options.lambda_max)
                    if abs(current_lambda - self.options.lambda_max) < 1e-9 : 
                        if self.options.verbose: print("Lambda reached max after linalg error, solver may be stuck.")
                        break 
                    continue # Retry with new lambda
                except RuntimeError as e: # Catches other runtime errors from solve, e.g. CUDA errors
                    if self.options.verbose: print(f"Runtime error during linalg.solve: {e} at iter {i}. Increasing lambda.")
                    current_lambda = min(current_lambda * self.options.lambda_factor, self.options.lambda_max)
                    if abs(current_lambda - self.options.lambda_max) < 1e-9:
                        if self.options.verbose: print("Lambda reached max after runtime error, solver may be stuck.")
                        break
                    continue
            
            step_norm = torch.linalg.norm(delta_tangent).item() if delta_tangent.numel() > 0 else 0.0

            if delta_tangent.numel() == 0 and self.problem.total_tangent_dim > 0:
                # No step possible even if variables exist (e.g. JTJ+lambda*I singular and neg_JTr is 0)
                if self.options.verbose: print(f"No step taken at iter {i} (delta_tangent is empty). Grad norm: {current_grad_norm:.2e}. Stopping.")
                break # Or check grad norm and continue if not met?

            # rho = actual_reduction / predicted_reduction
            # actual_reduction = current_cost - proposed_cost
            # predicted_reduction for model L(delta) = L(0) - delta^T * (-JTr) - 0.5 * delta^T * JTJ * delta
            # L(0) = 0.5 * r^T r (if using 0.5 factor) or r^T r (if not).
            # If cost = r^T r, then L(delta) approx current_cost + delta^T * (2 J^T r) + 0.5 * delta^T * (2 J^T J) * delta
            # Here, problem.build_system returns cost = r^T r, and neg_JTr = -J^T r.
            # Predicted reduction is current_cost - ( current_cost - delta^T(-neg_JTr) + 0.5 delta^T JTJ delta)
            # = delta^T(-neg_JTr) - 0.5 delta^T JTJ delta = delta^T (J^T r) - 0.5 delta^T JTJ delta
            # Since neg_JTr = -J^T r, then -neg_JTr = J^T r.
            # So predicted_reduction = delta_tangent @ (-neg_JTr) - 0.5 * delta_tangent @ JTJ @ delta_tangent (WRONG - sign error in first term)
            # Should be: predicted_reduction = - (delta_tangent @ neg_JTr + 0.5 * delta_tangent @ JTJ @ delta_tangent)
            # Or, if we define gain as positive: L(0) - L(h) = -[g^T h + 0.5 h^T H h]
            # Here g = J^T r = -neg_JTr. H = JTJ. h = delta_tangent.
            # Predicted reduction = -[(-neg_JTr)^T delta_tangent + 0.5 delta_tangent^T JTJ delta_tangent]
            # = neg_JTr^T delta_tangent - 0.5 delta_tangent^T JTJ delta_tangent (if neg_JTr is -JTr).
            # The original script had: predicted_reduction = delta_tangent @ (neg_JTr) - 0.5 * delta_tangent @ JTJ @ delta_tangent
            # This is correct if neg_JTr is used as -J^T r.
            
            if step_norm < self.options.tolerance_step_norm and i > 0: # If step is too small (and not first iter)
                # Check grad_norm before declaring convergence on small step.
                if current_grad_norm < self.options.tolerance_grad_norm:
                    if self.options.verbose: print(f"Converged: Step norm ({step_norm:.2e}) and grad norm ({current_grad_norm:.2e}) below tolerance.")
                    break
                else: # Step is small but gradient is not, could be stuck
                    if self.options.verbose: print(f"Step norm ({step_norm:.2e}) small, but grad norm ({current_grad_norm:.2e}) not. Increasing lambda.")
                    current_lambda = min(current_lambda * self.options.lambda_factor, self.options.lambda_max)
                    if abs(current_lambda - self.options.lambda_max) < 1e-9:
                        if self.options.verbose: print("Lambda reached max, step norm small but grad high. Stopping.")
                        break
                    continue

            proposed_values = self.problem.update_values(current_values, delta_tangent)
            _JTJ_prop, _neg_JTr_prop, proposed_cost = self.problem.build_system(proposed_values)

            if not math.isfinite(proposed_cost): 
                if self.options.verbose: print(f"Proposed cost is {proposed_cost} at iter {i}. Increasing lambda.")
                current_lambda = min(current_lambda * self.options.lambda_factor, self.options.lambda_max)
                if abs(current_lambda - self.options.lambda_max) < 1e-9 : 
                    if self.options.verbose: print("Lambda reached max after NaN cost, solver may be stuck.")
                    break
                continue # Retry with new lambda (and same current_values, JTJ, neg_JTr, current_cost)

            actual_reduction = current_cost - proposed_cost
            # Original predicted_reduction: delta_tangent @ (neg_JTr) - 0.5 * delta_tangent @ JTJ @ delta_tangent
            # This seems to be using neg_JTr as g_k (gradient), and JTJ as H_k (Hessian approx)
            # The linear model is m(p) = f_k + g_k^T p + 0.5 p^T H_k p
            # Predicted reduction f_k - m(p) = - (g_k^T p + 0.5 p^T H_k p)
            # If neg_JTr is -g_k, then g_k = -neg_JTr. 
            # Predicted reduction = - ((-neg_JTr)^T delta_tangent + 0.5 delta_tangent^T JTJ delta_tangent)
            #                  = neg_JTr^T delta_tangent - 0.5 delta_tangent^T JTJ delta_tangent
            # This matches the original script if neg_JTr is literally taken as the vector g in the formula for pred_red.
            # Given neg_JTr from build_system is -J^T r, this formula is: 
            # delta_tangent @ (-J^T r) - 0.5 * delta_tangent @ JTJ @ delta_tangent
            # Which is the standard predicted reduction (with a negative sign if reduction is positive gain).
            # Let's assume predicted_reduction is positive for actual reduction.
            # So, pred_red = - ( delta^T g + 0.5 delta^T H delta ) where g = J^T r = -neg_JTr
            # pred_red = - ( delta^T (-neg_JTr) + 0.5 delta^T JTJ delta) = delta^T neg_JTr - 0.5 delta^T JTJ delta
            predicted_reduction = (delta_tangent @ neg_JTr - 0.5 * delta_tangent @ (JTJ @ delta_tangent)) if delta_tangent.numel() > 0 else 0.0
            
            rho = actual_reduction / (predicted_reduction + 1e-12) if predicted_reduction != 0 else float('inf') * torch.sign(actual_reduction)
            if predicted_reduction == 0 and actual_reduction == 0: rho = 1.0 # Avoid 0/0, treat as perfect prediction
            if predicted_reduction == 0 and actual_reduction != 0: rho = float('inf') * torch.sign(actual_reduction)

            if self.options.verbose:
                print(f"{i:4} | {current_cost:12.6e} | {current_lambda:10.3e} | {rho:10.3f} | {step_norm:12.6e} | {current_grad_norm:12.6e}")

            if rho > 0.01: # Step is good or acceptable
                cost_delta_abs = current_cost - proposed_cost 
                cost_delta_rel = cost_delta_abs / (abs(current_cost) + 1e-12) if abs(current_cost) > 1e-12 else cost_delta_abs
                
                current_values = proposed_values
                JTJ, neg_JTr, current_cost = _JTJ_prop, _neg_JTr_prop, proposed_cost # Update system state
                current_grad_norm = torch.linalg.norm(neg_JTr).item() if neg_JTr.numel() > 0 else 0.0

                # Check for convergence after a successful step
                cost_converged = (abs(cost_delta_abs) < self.options.tolerance_cost_delta_abs or \
                                  abs(cost_delta_rel) < self.options.tolerance_cost_delta_rel)
                # Step norm already calculated as step_norm
                if cost_converged and step_norm < self.options.tolerance_step_norm:
                    if self.options.verbose: print("Converged: Cost change and step norm below tolerance.")
                    break
                if current_grad_norm < self.options.tolerance_grad_norm:
                     if self.options.verbose: print("Converged: Gradient norm below tolerance.")
                     break
                
                # Update lambda (decrease for good step)
                current_lambda = max(self.options.lambda_min, current_lambda * max(1/3, 1 - (2*rho - 1)**3))
            else: # Step is bad (rho <= 0.01)
                # Increase lambda (make it more like gradient descent)
                current_lambda = min(current_lambda * self.options.lambda_factor, self.options.lambda_max)
                if abs(current_lambda - self.options.lambda_max) < 1e-9 and rho <= 0: # Check if lambda is maxed out
                    if self.options.verbose: print("Lambda reached max and step quality is poor. Optimization may be stuck.")
                    break
            
            if i == self.options.max_iterations - 1 :
                 if self.options.verbose: print("Reached max iterations.")
        
        return current_values 