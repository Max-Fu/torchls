# Top-level __init__.py for torchls package

# Import key classes and functions to make them available at the top level of the package

# Core components
from .core.problem import LeastSquaresProblem
from .core.cost import Cost

# Variables
from .variables.base import Variable
from .variables.lie_groups import LieGroupVariable, SE3Variable

# Solvers
from .solvers.options import SolverOptions
from .solvers.gauss_newton import GaussNewtonSolver
from .solvers.lm import LevenbergMarquardtSolver

# Lie Math (if users need direct access)
from .lie_math.se3 import se3_exp_map, se3_log_map

# Utilities (DEVICE, DTYPE are set globally but can be exposed if needed)
from .utils.misc import DEVICE, DEFAULT_DTYPE

__all__ = [
    "LeastSquaresProblem", "Cost", 
    "Variable", "LieGroupVariable", "SE3Variable",
    "SolverOptions", "GaussNewtonSolver", "LevenbergMarquardtSolver",
    "se3_exp_map", "se3_log_map",
    "DEVICE", "DEFAULT_DTYPE"
]

__version__ = "0.1.0" # Example version 