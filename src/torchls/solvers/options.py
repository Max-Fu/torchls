class SolverOptions:
    """
    Configuration options for optimization solvers.

    Args:
        max_iterations (int): Maximum number of iterations for the solver.
        lambda_init (float): Initial damping factor for Levenberg-Marquardt.
        lambda_factor (float): Factor to increase/decrease lambda in Levenberg-Marquardt.
        lambda_max (float): Maximum allowable value for lambda.
        lambda_min (float): Minimum allowable value for lambda.
        tolerance_cost_delta_abs (float): Absolute tolerance for change in cost function value.
        tolerance_cost_delta_rel (float): Relative tolerance for change in cost function value.
        tolerance_step_norm (float): Tolerance for the norm of the update step (delta_x).
        tolerance_grad_norm (float): Tolerance for the norm of the gradient (J^T * r).
        verbose (bool): If True, print optimization progress.
    """
    def __init__(self, max_iterations: int = 20,
                 lambda_init: float = 1e-3, lambda_factor: float = 10.0,
                 lambda_max: float = 1e6, lambda_min: float = 1e-9,
                 tolerance_cost_delta_abs: float = 1e-7, 
                 tolerance_cost_delta_rel: float = 1e-7, 
                 tolerance_step_norm: float = 1e-7,
                 tolerance_grad_norm: float = 1e-7, verbose: bool = False):
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.tolerance_cost_delta_abs = tolerance_cost_delta_abs
        self.tolerance_cost_delta_rel = tolerance_cost_delta_rel
        self.tolerance_step_norm = tolerance_step_norm
        self.tolerance_grad_norm = tolerance_grad_norm
        self.verbose = verbose 