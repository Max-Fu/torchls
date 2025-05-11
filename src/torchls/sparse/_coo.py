import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SparseCooMatrix:
    """
    Represents a sparse matrix in Coordinate (COO) format.

    Attributes:
        values (torch.Tensor): A 1D tensor containing the non-zero values of the sparse matrix.
            Shape: (nnz,)
        row_indices (torch.Tensor): A 1D tensor containing the row indices of the non-zero values.
            Shape: (nnz,)
        col_indices (torch.Tensor): A 1D tensor containing the column indices of the non-zero values.
            Shape: (nnz,)
        shape (Tuple[int, int]): A tuple representing the dimensions (rows, cols) of the sparse matrix.
    """
    values: torch.Tensor
    row_indices: torch.Tensor
    col_indices: torch.Tensor
    shape: Tuple[int, int]

    def __post_init__(self):
        # Basic validation
        if not (self.values.ndim == 1 and 
                self.row_indices.ndim == 1 and 
                self.col_indices.ndim == 1):
            raise ValueError("values, row_indices, and col_indices must be 1D tensors.")
        if not (self.values.shape[0] == self.row_indices.shape[0] == self.col_indices.shape[0]):
            raise ValueError("values, row_indices, and col_indices must have the same length (nnz).")
        if not (len(self.shape) == 2 and self.shape[0] > 0 and self.shape[1] > 0):
            raise ValueError("Shape must be a 2-tuple of positive integers (rows, cols).")
        if self.row_indices.numel() > 0: # Check only if there are non-zero elements
            if not (self.row_indices.max() < self.shape[0] and self.col_indices.max() < self.shape[1]):
                raise ValueError("Row/column indices are out of bounds for the given shape.")

    def to_dense(self) -> torch.Tensor:
        """
        Converts the sparse COO matrix to a dense PyTorch tensor.

        Returns:
            torch.Tensor: The dense representation of the matrix.
                Shape: (self.shape[0], self.shape[1])
        """
        return torch.sparse_coo_tensor(
            indices=torch.stack([self.row_indices, self.col_indices]),
            values=self.values,
            size=self.shape,
            device=self.values.device, # Ensure device consistency
            dtype=self.values.dtype   # Ensure dtype consistency
        ).to_dense()

    def to_torch_sparse_coo(self) -> torch.Tensor:
        """
        Converts this SparseCooMatrix to a PyTorch sparse COO tensor.

        Returns:
            torch.Tensor: A PyTorch sparse COO tensor.
        """
        return torch.sparse_coo_tensor(
            indices=torch.stack([self.row_indices, self.col_indices]),
            values=self.values,
            size=self.shape,
            device=self.values.device,
            dtype=self.values.dtype
        )

    @classmethod
    def from_torch_sparse_coo(cls, tensor: torch.Tensor) -> 'SparseCooMatrix':
        """
        Creates a SparseCooMatrix from a PyTorch sparse COO tensor.

        Args:
            tensor (torch.Tensor): A PyTorch sparse COO tensor. 
                Must be coalesced.

        Returns:
            SparseCooMatrix: An instance of SparseCooMatrix.
        
        Raises:
            ValueError: If the input tensor is not a sparse COO tensor.
        """
        if tensor.layout != torch.sparse_coo:
            raise ValueError("Input tensor must be a PyTorch sparse COO tensor.")
        
        tensor_coalesced = tensor.coalesce()
        
        values = tensor_coalesced.values()
        indices = tensor_coalesced.indices()
        row_indices = indices[0]
        col_indices = indices[1]
        shape = tuple(tensor_coalesced.shape)
        
        return cls(values, row_indices, col_indices, shape)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"values={self.values}, "
                f"row_indices={self.row_indices}, "
                f"col_indices={self.col_indices}, "
                f"shape={self.shape})")

# Example Usage (can be moved to a test file later)
if __name__ == '__main__':
    # Create a SparseCooMatrix
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    r = torch.tensor([0, 0, 1, 2, 2])
    c = torch.tensor([0, 2, 1, 0, 2])
    s = (3, 4)
    
    coo_matrix = SparseCooMatrix(values=v, row_indices=r, col_indices=c, shape=s)
    print("Custom SparseCooMatrix:\n", coo_matrix)

    # Convert to dense
    dense_matrix = coo_matrix.to_dense()
    print("\nDense Matrix:\n", dense_matrix)

    # Convert to PyTorch sparse COO tensor
    torch_sparse_coo = coo_matrix.to_torch_sparse_coo()
    print("\nPyTorch Sparse COO Tensor:\n", torch_sparse_coo)
    print("\nPyTorch Sparse COO Tensor (dense view):\n", torch_sparse_coo.to_dense())


    # Create from PyTorch sparse COO tensor
    # Ensure indices are 2xN for torch.sparse_coo_tensor
    indices_torch = torch.tensor([[0, 0, 1, 2, 2],
                                  [0, 2, 1, 0, 2]]) 
    values_torch = torch.tensor([10., 20., 30., 40., 50.])
    shape_torch = (3, 4)
    
    torch_sparse_tensor = torch.sparse_coo_tensor(indices_torch, values_torch, shape_torch)
    print("\nOriginal PyTorch Sparse Tensor (coalesced):\n", torch_sparse_tensor.coalesce())

    coo_from_torch = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_tensor)
    print("\nCustom SparseCooMatrix from PyTorch sparse tensor:\n", coo_from_torch)
    print("\nDense from custom (created from PyTorch sparse):\n", coo_from_torch.to_dense())

    # Test validation
    try:
        SparseCooMatrix(torch.tensor([[1.],[2.]]), r, c, s) # Wrong values dim
    except ValueError as e:
        print(f"\nCaught expected error: {e}")

    try:
        SparseCooMatrix(v, torch.tensor([0,0,1,2,3]), c, s) # Row index out of bounds
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
        
    # Example of an empty sparse matrix
    empty_coo = SparseCooMatrix(
        values=torch.empty(0, dtype=torch.float300),
        row_indices=torch.empty(0, dtype=torch.long),
        col_indices=torch.empty(0, dtype=torch.long),
        shape=(5,5)
    )
    print("\nEmpty SparseCooMatrix:\n", empty_coo)
    print("\nEmpty SparseCooMatrix (dense):\n", empty_coo.to_dense())
    print("\nEmpty SparseCooMatrix (torch sparse coo):\n", empty_coo.to_torch_sparse_coo())

    # Check device and dtype transfer
    if torch.cuda.is_available():
        v_cuda = v.cuda()
        r_cuda = r.cuda()
        c_cuda = c.cuda()
        coo_cuda = SparseCooMatrix(v_cuda, r_cuda, c_cuda, s)
        torch_sparse_cuda = coo_cuda.to_torch_sparse_coo()
        print(f"\nCUDA COO matrix device: {coo_cuda.values.device}")
        print(f"\nCUDA torch_sparse_coo device: {torch_sparse_cuda.device}")
        assert torch_sparse_cuda.device.type == 'cuda'

        coo_from_torch_cuda = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_cuda)
        print(f"\nCUDA coo_from_torch_cuda device: {coo_from_torch_cuda.values.device}")
        assert coo_from_torch_cuda.values.device.type == 'cuda'
        
    v_float64 = v.to(torch.float64)
    coo_float64 = SparseCooMatrix(v_float64, r, c, s)
    torch_sparse_float64 = coo_float64.to_torch_sparse_coo()
    print(f"\nFloat64 COO matrix dtype: {coo_float64.values.dtype}")
    print(f"\nFloat64 torch_sparse_coo dtype: {torch_sparse_float64.dtype}")
    assert torch_sparse_float64.dtype == torch.float64

    coo_from_torch_f64 = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_float64)
    print(f"\nFloat64 coo_from_torch_f64 dtype: {coo_from_torch_f64.values.dtype}")
    assert coo_from_torch_f64.values.dtype == torch.float64 