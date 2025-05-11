import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SparseCsrMatrix:
    """
    Represents a sparse matrix in Compressed Sparse Row (CSR) format.

    Attributes:
        values (torch.Tensor): A 1D tensor containing the non-zero values of the sparse matrix,
            ordered row by row. Shape: (nnz,)
        col_indices (torch.Tensor): A 1D tensor containing the column indices of the non-zero values.
            Shape: (nnz,)
        crow_indices (torch.Tensor): A 1D tensor representing the compressed row pointers.
            It indicates the start of each row's non-zero elements in the `values` and `col_indices` tensors.
            Shape: (num_rows + 1,)
        shape (Tuple[int, int]): A tuple representing the dimensions (rows, cols) of the sparse matrix.
    """
    values: torch.Tensor
    col_indices: torch.Tensor
    crow_indices: torch.Tensor # Compressed Row Indices (like indptr)
    shape: Tuple[int, int]

    def __post_init__(self):
        # Basic validation
        if not (self.values.ndim == 1 and 
                self.col_indices.ndim == 1 and 
                self.crow_indices.ndim == 1):
            raise ValueError("values, col_indices, and crow_indices must be 1D tensors.")
        
        if not (self.values.shape[0] == self.col_indices.shape[0]):
            raise ValueError("values and col_indices must have the same length (nnz).")

        if not (len(self.shape) == 2 and self.shape[0] >= 0 and self.shape[1] >= 0):
            raise ValueError("Shape must be a 2-tuple of non-negative integers (rows, cols).")
        
        if not (self.crow_indices.shape[0] == self.shape[0] + 1):
            raise ValueError(f"crow_indices must have length num_rows + 1 ({self.shape[0] + 1}), got {self.crow_indices.shape[0]}.")

        if self.values.numel() > 0: # Check only if there are non-zero elements
            if not (self.col_indices.max() < self.shape[1]):
                raise ValueError("Column indices are out of bounds for the given shape.")
            if not (self.crow_indices[0] == 0):
                raise ValueError("First element of crow_indices must be 0.")
            if not (self.crow_indices[-1] == self.values.shape[0]):
                raise ValueError(f"Last element of crow_indices must be nnz ({self.values.shape[0]}), got {self.crow_indices[-1]}.")
            if not torch.all(self.crow_indices[:-1] <= self.crow_indices[1:]):
                 raise ValueError("crow_indices must be monotonically non-decreasing.")
        elif self.shape[0] > 0: # nnz is 0, but matrix has rows
             if not (self.crow_indices[0] == 0 and self.crow_indices[-1] == 0 and torch.all(self.crow_indices == 0)):
                 raise ValueError("For an empty matrix, crow_indices must be all zeros if num_rows > 0.")
        elif self.shape[0] == 0: # 0 rows, 0 nnz
            if not (self.crow_indices.shape[0] == 1 and self.crow_indices[0] == 0):
                raise ValueError("For a matrix with 0 rows, crow_indices must be a tensor containing a single 0.")


    def to_dense(self) -> torch.Tensor:
        """
        Converts the sparse CSR matrix to a dense PyTorch tensor.

        Returns:
            torch.Tensor: The dense representation of the matrix.
                Shape: (self.shape[0], self.shape[1])
        """
        if self.shape == (0,0) and self.values.numel() == 0:
             return torch.empty((0,0), dtype=self.values.dtype, device=self.values.device)
        return torch.sparse_csr_tensor(
            crow_indices=self.crow_indices,
            col_indices=self.col_indices,
            values=self.values,
            size=self.shape,
            device=self.values.device, # Ensure device consistency
            dtype=self.values.dtype   # Ensure dtype consistency
        ).to_dense()

    def to_torch_sparse_csr(self) -> torch.Tensor:
        """
        Converts this SparseCsrMatrix to a PyTorch sparse CSR tensor.

        Returns:
            torch.Tensor: A PyTorch sparse CSR tensor.
        """
        if self.shape == (0,0) and self.values.numel() == 0:
            # torch.sparse_csr_tensor for (0,0) shape needs special handling for consistency
            # depending on PyTorch version. Often an empty tensor of the right type is fine.
            # However, creating from components for (0,0) can be tricky.
            # Let's assume for now that if shape is (0,0) and nnz is 0, an empty sparse tensor is intended.
            # This might need refinement based on specific torch version behaviors.
            return torch.sparse_csr_tensor(
                torch.tensor([0], dtype=self.crow_indices.dtype, device=self.crow_indices.device),
                torch.empty((0), dtype=self.col_indices.dtype, device=self.col_indices.device),
                torch.empty((0), dtype=self.values.dtype, device=self.values.device),
                size=(0,0)
            )

        return torch.sparse_csr_tensor(
            crow_indices=self.crow_indices,
            col_indices=self.col_indices,
            values=self.values,
            size=self.shape,
            device=self.values.device,
            dtype=self.values.dtype
        )

    @classmethod
    def from_torch_sparse_csr(cls, tensor: torch.Tensor) -> 'SparseCsrMatrix':
        """
        Creates a SparseCsrMatrix from a PyTorch sparse CSR tensor.

        Args:
            tensor (torch.Tensor): A PyTorch sparse CSR tensor.

        Returns:
            SparseCsrMatrix: An instance of SparseCsrMatrix.
        
        Raises:
            ValueError: If the input tensor is not a sparse CSR tensor.
        """
        if not tensor.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor.")
        
        values = tensor.values()
        col_indices = tensor.col_indices()
        crow_indices = tensor.crow_indices()
        shape = tuple(tensor.shape)
        
        return cls(values, col_indices, crow_indices, shape)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"values={self.values}, "
                f"col_indices={self.col_indices}, "
                f"crow_indices={self.crow_indices}, "
                f"shape={self.shape})")

# Example Usage (can be moved to a test file later)
if __name__ == '__main__':
    # Create a SparseCsrMatrix
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = torch.tensor([0, 2, 1, 0, 2]) # col_indices
    cr = torch.tensor([0, 2, 3, 5])    # crow_indices (num_rows + 1 = 3 + 1 = 4)
    s = (3, 4) # num_rows = 3, num_cols = 4
    
    csr_matrix = SparseCsrMatrix(values=v, col_indices=ci, crow_indices=cr, shape=s)
    print("Custom SparseCsrMatrix:\n", csr_matrix)

    # Convert to dense
    dense_matrix = csr_matrix.to_dense()
    print("\nDense Matrix:\n", dense_matrix)
    # Expected dense matrix:
    # [[1., 0., 2., 0.],
    #  [0., 3., 0., 0.],
    #  [4., 0., 5., 0.]]

    # Convert to PyTorch sparse CSR tensor
    torch_sparse_csr = csr_matrix.to_torch_sparse_csr()
    print("\nPyTorch Sparse CSR Tensor:\n", torch_sparse_csr)
    print("\nPyTorch Sparse CSR Tensor (dense view):\n", torch_sparse_csr.to_dense())

    # Create from PyTorch sparse CSR tensor
    torch_csr_tensor = torch.sparse_csr_tensor(cr, ci, v, s)
    print("\nOriginal PyTorch Sparse CSR Tensor:\n", torch_csr_tensor)

    csr_from_torch = SparseCsrMatrix.from_torch_sparse_csr(torch_csr_tensor)
    print("\nCustom SparseCsrMatrix from PyTorch sparse tensor:\n", csr_from_torch)
    print("\nDense from custom (created from PyTorch sparse):\n", csr_from_torch.to_dense())

    # Test validation
    try:
        # Wrong crow_indices length
        SparseCsrMatrix(v, ci, torch.tensor([0, 2, 3]), s)
    except ValueError as e:
        print(f"\nCaught expected error for crow_indices length: {e}")

    try:
        # crow_indices does not start with 0
        SparseCsrMatrix(v, ci, torch.tensor([1, 2, 3, 5]), s)
    except ValueError as e:
        print(f"\nCaught expected error for crow_indices start: {e}")

    try:
        # crow_indices does not end with nnz
        SparseCsrMatrix(v, ci, torch.tensor([0, 2, 3, 4]), s)
    except ValueError as e:
        print(f"\nCaught expected error for crow_indices end: {e}")
        
    try:
        # crow_indices not non-decreasing
        SparseCsrMatrix(v, ci, torch.tensor([0, 3, 2, 5]), s)
    except ValueError as e:
        print(f"\nCaught expected error for crow_indices monotonicity: {e}")

    # Example of an empty sparse matrix (0x0)
    empty_csr_0x0 = SparseCsrMatrix(
        values=torch.empty(0, dtype=torch.float32),
        col_indices=torch.empty(0, dtype=torch.long),
        crow_indices=torch.tensor([0], dtype=torch.long),
        shape=(0,0)
    )
    print("\nEmpty 0x0 SparseCsrMatrix:\n", empty_csr_0x0)
    print("\nEmpty 0x0 SparseCsrMatrix (dense):\n", empty_csr_0x0.to_dense())
    print("\nEmpty 0x0 SparseCsrMatrix (torch sparse csr):\n", empty_csr_0x0.to_torch_sparse_csr().to_dense())

    # Example of an empty sparse matrix (3x4 with 0 nnz)
    empty_csr_3x4_nnz0 = SparseCsrMatrix(
        values=torch.empty(0, dtype=torch.float32),
        col_indices=torch.empty(0, dtype=torch.long),
        crow_indices=torch.tensor([0, 0, 0, 0], dtype=torch.long), # num_rows = 3 -> len = 4
        shape=(3,4)
    )
    print("\nEmpty 3x4 (0 nnz) SparseCsrMatrix:\n", empty_csr_3x4_nnz0)
    print("\nEmpty 3x4 (0 nnz) SparseCsrMatrix (dense):\n", empty_csr_3x4_nnz0.to_dense())
    print("\nEmpty 3x4 (0 nnz) SparseCsrMatrix (torch sparse csr):\n", empty_csr_3x4_nnz0.to_torch_sparse_csr().to_dense()) 

    # Test device and dtype
    if torch.cuda.is_available():
        v_cuda = v.cuda()
        ci_cuda = ci.cuda()
        cr_cuda = cr.cuda()
        csr_cuda = SparseCsrMatrix(v_cuda, ci_cuda, cr_cuda, s)
        torch_sparse_cuda = csr_cuda.to_torch_sparse_csr()
        print(f"\nCUDA CSR matrix device: {csr_cuda.values.device}")
        print(f"\nCUDA torch_sparse_csr device: {torch_sparse_cuda.device}")
        assert torch_sparse_cuda.device.type == 'cuda'

        csr_from_torch_cuda = SparseCsrMatrix.from_torch_sparse_csr(torch_sparse_cuda)
        print(f"\nCUDA csr_from_torch_cuda device: {csr_from_torch_cuda.values.device}")
        assert csr_from_torch_cuda.values.device.type == 'cuda'
        
    v_float64 = v.to(torch.float64)
    csr_float64 = SparseCsrMatrix(v_float64, ci, cr, s)
    torch_sparse_float64 = csr_float64.to_torch_sparse_csr()
    print(f"\nFloat64 CSR matrix dtype: {csr_float64.values.dtype}")
    print(f"\nFloat64 torch_sparse_csr dtype: {torch_sparse_float64.dtype}")
    assert torch_sparse_float64.dtype == torch.float64

    coo_from_torch_f64 = SparseCsrMatrix.from_torch_sparse_csr(torch_sparse_float64)
    print(f"\nFloat64 coo_from_torch_f64 dtype: {coo_from_torch_f64.values.dtype}")
    assert coo_from_torch_f64.values.dtype == torch.float64 