import torch
import abc
from typing import Optional, Union, Tuple

from .base import Variable
from ..lie_math.se3 import se3_exp_map, se3_log_map
from ..utils.misc import DEVICE, DEFAULT_DTYPE

class LieGroupVariable(Variable, abc.ABC):
    """
    Abstract base class for Lie group variables.

    Args:
        initial_value (torch.Tensor): The initial value of the Lie group variable.
                                      Shape depends on the specific Lie group.
        name (str, optional): Name of the variable.

    Attributes:
        initial_value (torch.Tensor): Stores the initial state of the variable on the correct device and dtype.
    """
    def __init__(self, initial_value: torch.Tensor, name: str = ""):
        super().__init__(name)
        assert initial_value.ndim >= 1, "Initial value must be at least a 1D tensor."
        self.initial_value = initial_value.to(device=DEVICE, dtype=DEFAULT_DTYPE)

    @property
    @abc.abstractmethod
    def tangent_dim(self) -> int:
        """int: The dimension of the tangent space for this Lie group."""
        pass

    @abc.abstractmethod
    def retract(self, current_value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Performs the retraction operation (exponential map or similar) to update the variable.

        Args:
            current_value (torch.Tensor): The current value of the Lie group variable.
                                          Shape (..., D_manifold) or (D_manifold).
            delta (torch.Tensor): The update in the tangent space.
                                  Shape (..., D_tangent) or (D_tangent).

        Returns:
            torch.Tensor: The updated Lie group variable.
                          Shape matches `current_value` if delta is unbatched, or `delta` if `current_value` is unbatched.
                          If both are batched, their batch dimensions must be compatible.
        """
        pass

    @abc.abstractmethod
    def local_coordinates(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        """
        Computes the local coordinates (logarithm map or similar) between two variable values.

        Args:
            value1 (torch.Tensor): The first Lie group variable value.
                                   Shape (..., D_manifold) or (D_manifold).
            value2 (torch.Tensor): The second Lie group variable value.
                                   Shape (..., D_manifold) or (D_manifold).

        Returns:
            torch.Tensor: The difference in the tangent space.
                          Shape (..., D_tangent) or (D_tangent), matching batching of inputs.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def identity(cls, batch_size: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
        """
        Returns the identity element for this Lie group.

        Args:
            batch_size (Optional[Union[int, Tuple]], optional):
                If provided and > 0, returns a batch of identity elements.
                If an empty tuple or 0, returns a single unbatched identity element.
                If None, returns a single identity element wrapped in a batch of size 1.
                Defaults to None.

        Returns:
            torch.Tensor: The identity element(s).
        """
        pass

class SE3Variable(LieGroupVariable):
    """
    Represents a variable in the SE(3) Lie group (3D rigid body transformations).
    The internal representation is a batch of 4x4 transformation matrices.

    Args:
        initial_value (Optional[torch.Tensor], optional):
            Initial value as a (4,4) matrix or a batch of (B,4,4) matrices.
            If None, defaults to the identity transformation. Defaults to None.
        name (str, optional): Name of the variable. Defaults to "".
    """
    _tangent_dim = 6

    def __init__(self, initial_value: Optional[torch.Tensor] = None, name: str = ""):
        if initial_value is None:
            initial_value = self.identity() # Get (1,4,4) identity
        elif initial_value.ndim == 2 and initial_value.shape == (4, 4):
            initial_value = initial_value.unsqueeze(0) # Internally store as (1,4,4)
        
        assert initial_value.ndim == 3 and initial_value.shape[-2:] == (4,4), \
            f"SE3 initial_value must be (B,4,4) or (4,4), got {initial_value.shape}"
        super().__init__(initial_value, name)

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim
    
    def retract(self, current_value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Updates an SE(3) transformation by applying a tangent space update (delta).
        T_new = exp(delta) * T_current

        Args:
            current_value (torch.Tensor): Current SE(3) value(s), shape (B,4,4) or (4,4).
            delta (torch.Tensor): Tangent space update(s), shape (B,6) or (6).

        Returns:
            torch.Tensor: Updated SE(3) value(s), shape matches batched input conventions.
        """
        input_current_batched = current_value.ndim == 3
        input_delta_batched = delta.ndim == 2

        current_value_b = current_value if input_current_batched else current_value.unsqueeze(0)
        delta_b = delta if input_delta_batched else delta.unsqueeze(0)

        if delta_b.shape[0] != current_value_b.shape[0]:
            if delta_b.shape[0] == 1 and current_value_b.shape[0] > 1:
                delta_b = delta_b.expand(current_value_b.shape[0], -1)
            elif current_value_b.shape[0] == 1 and delta_b.shape[0] > 1:
                 current_value_b = current_value_b.expand(delta_b.shape[0],-1,-1)
            else: # Mismatched batch sizes > 1
                 raise ValueError(f"Batch dimensions of delta {delta.shape} and current_value {current_value.shape} are incompatible.")
        
        # delta_transform is (B,4,4), current_value_b is (B,4,4)
        delta_transform = se3_exp_map(delta_b) 
        retracted_val = torch.matmul(delta_transform, current_value_b)
        
        # Determine output batching: if original current_value was unbatched and delta was singular or matched to unbatched current,
        # and the result is (1,4,4), then squeeze.
        if not input_current_batched and retracted_val.shape[0] == 1:
            return retracted_val.squeeze(0)
        return retracted_val

    def local_coordinates(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        """
        Computes the tangent space vector from value1 to value2 in SE(3).
        delta = log(value1_inv * value2)

        Args:
            value1 (torch.Tensor): Starting SE(3) value(s), shape (B,4,4) or (4,4).
            value2 (torch.Tensor): Ending SE(3) value(s), shape (B,4,4) or (4,4).

        Returns:
            torch.Tensor: Tangent space vector(s), shape (B,6) or (6).
        """
        input_v1_batched = value1.ndim == 3
        input_v2_batched = value2.ndim == 3

        value1_b = value1 if input_v1_batched else value1.unsqueeze(0)
        value2_b = value2 if input_v2_batched else value2.unsqueeze(0)

        if value1_b.shape[0] != value2_b.shape[0]:
            if value1_b.shape[0] == 1 and value2_b.shape[0] > 1: value1_b = value1_b.expand(value2_b.shape[0],-1,-1)
            elif value2_b.shape[0] == 1 and value1_b.shape[0] > 1: value2_b = value2_b.expand(value1_b.shape[0],-1,-1)
            else:
                 raise ValueError(f"Batch dimensions of value1 {value1.shape} and value2 {value2.shape} are incompatible.")

        value1_inv = torch.linalg.inv(value1_b) # (B,4,4)
        T_ab = torch.matmul(value2_b, value1_inv) # (B,4,4)  value2 @ value1.inv
        log_map_result = se3_log_map(T_ab) # (B,6)
        
        # Determine output batching
        if not input_v1_batched and not input_v2_batched and log_map_result.shape[0] == 1:
            return log_map_result.squeeze(0)
        return log_map_result
    
    @classmethod
    def identity(cls, batch_size: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
        """
        Returns the identity 4x4 SE(3) transformation matrix.

        Args:
            batch_size (Optional[Union[int, Tuple]], optional):
                If None, returns a (1,4,4) identity tensor.
                If an int > 0, returns a (batch_size, 4,4) tensor.
                If an empty tuple or 0, returns a (4,4) tensor.
                Defaults to None.

        Returns:
            torch.Tensor: The SE(3) identity matrix/matrices.
        """
        id_mat = torch.eye(4, device=DEVICE, dtype=DEFAULT_DTYPE)
        if batch_size is None: 
            return id_mat.unsqueeze(0) # Default: (1,4,4)
        if isinstance(batch_size, tuple) and not batch_size: # batch_size=() -> (4,4)
            return id_mat 
        if isinstance(batch_size, int):
            if batch_size == 0: # batch_size=0 -> (4,4)
                 return id_mat
            elif batch_size > 0: # batch_size=N -> (N,4,4)
                return id_mat.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Fallback for other tuple cases or unhandled int values, though primarily covered.
        return id_mat.unsqueeze(0) # Default to (1,4,4) 