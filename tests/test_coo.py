import torch
import unittest
from torchls.sparse import SparseCooMatrix # Adjusted import path

class TestSparseCooMatrix(unittest.TestCase):

    def setUp(self):
        """Set up common data for tests."""
        self.v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        self.r = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
        self.c = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
        self.s = (3, 4)
        self.expected_dense = torch.tensor([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 5.0, 0.0]
        ], dtype=torch.float64)

        # Data for creating from torch sparse tensor
        indices_torch = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 2]], dtype=torch.long)
        values_torch = torch.tensor([10., 20., 30., 40., 50.], dtype=torch.float64)
        shape_torch = (3, 4)
        self.torch_sparse_tensor_alt = torch.sparse_coo_tensor(indices_torch, values_torch, shape_torch).coalesce()
        self.expected_dense_alt = torch.tensor([
            [10.0, 0.0, 20.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
            [40.0, 0.0, 50.0, 0.0]
        ], dtype=torch.float64)

    def test_coo_matrix_creation(self):
        """Tests basic creation of SparseCooMatrix."""
        coo_matrix = SparseCooMatrix(
            values=self.v,
            row_indices=self.r,
            col_indices=self.c,
            shape=self.s
        )
        self.assertTrue(torch.equal(coo_matrix.values, self.v))
        self.assertTrue(torch.equal(coo_matrix.row_indices, self.r))
        self.assertTrue(torch.equal(coo_matrix.col_indices, self.c))
        self.assertEqual(coo_matrix.shape, self.s)
        # print(f"Created SparseCooMatrix: {coo_matrix}") # Optional print

    def test_coo_to_dense(self):
        """Tests conversion of SparseCooMatrix to a dense tensor."""
        coo_matrix = SparseCooMatrix(
            values=self.v,
            row_indices=self.r,
            col_indices=self.c,
            shape=self.s
        )
        dense_matrix = coo_matrix.to_dense()
        self.assertTrue(torch.equal(dense_matrix, self.expected_dense))
        # print(f"Dense matrix from COO: \n{dense_matrix}") # Optional print

    def test_coo_to_torch_sparse_coo(self):
        """Tests conversion to PyTorch's sparse COO tensor."""
        coo_matrix = SparseCooMatrix(
            values=self.v,
            row_indices=self.r,
            col_indices=self.c,
            shape=self.s
        )
        torch_sparse_coo = coo_matrix.to_torch_sparse_coo()
        self.assertEqual(torch_sparse_coo.layout, torch.sparse_coo)
        self.assertTrue(torch.equal(torch_sparse_coo.to_dense(), self.expected_dense))
        
        torch_sparse_coo_coalesced = torch_sparse_coo.coalesce()
        self.assertTrue(torch.equal(torch_sparse_coo_coalesced.values(), self.v))
        expected_indices = torch.stack([self.r, self.c])
        self.assertTrue(torch.equal(torch_sparse_coo_coalesced.indices(), expected_indices))
        self.assertEqual(tuple(torch_sparse_coo_coalesced.shape), self.s)
        # print(f"PyTorch sparse COO tensor: \n{torch_sparse_coo}") # Optional print

    def test_coo_from_torch_sparse_coo(self):
        """Tests creation from PyTorch's sparse COO tensor."""
        torch_sparse_tensor = self.torch_sparse_tensor_alt
        coo_from_torch = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_tensor)
        
        self.assertTrue(torch.equal(coo_from_torch.values, torch_sparse_tensor.values()))
        self.assertTrue(torch.equal(coo_from_torch.row_indices, torch_sparse_tensor.indices()[0]))
        self.assertTrue(torch.equal(coo_from_torch.col_indices, torch_sparse_tensor.indices()[1]))
        self.assertEqual(coo_from_torch.shape, tuple(torch_sparse_tensor.shape))
        self.assertTrue(torch.equal(coo_from_torch.to_dense(), self.expected_dense_alt))
        # print(f"SparseCooMatrix from torch sparse: {coo_from_torch}") # Optional print

    def test_coo_validation_errors(self):
        """Tests that appropriate ValueErrors are raised for invalid inputs."""
        v, r, c, s = self.v, self.r, self.c, self.s

        with self.assertRaisesRegex(ValueError, "must be 1D tensors"):
            SparseCooMatrix(torch.tensor([[1.],[2.]]), r, c, s)
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            SparseCooMatrix(v[:-1], r, c, s)
        with self.assertRaisesRegex(ValueError, "Shape must be a 2-tuple"):
            SparseCooMatrix(v, r, c, (3,))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            SparseCooMatrix(v, torch.tensor([0,0,1,2,3]), c, s) # Row index 3 is out for shape (3,4)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            SparseCooMatrix(v, r, torch.tensor([0,0,1,0,4]), s) # Col index 4 is out for shape (3,4)

        # Test with non-coalesced tensor for from_torch_sparse_coo (it should coalesce it)
        indices_non_coalesced = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.long)
        values_non_coalesced = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        shape_non_coalesced = (2,2)
        torch_sparse_non_coalesced = torch.sparse_coo_tensor(indices_non_coalesced, values_non_coalesced, shape_non_coalesced)
        coo_from_non_coalesced = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_non_coalesced)
        
        self.assertTrue(torch.equal(coo_from_non_coalesced.to_dense(), torch_sparse_non_coalesced.to_dense()))
        internal_torch_sparse = coo_from_non_coalesced.to_torch_sparse_coo()
        # Explicitly coalesce before checking the flag, as the constructor might not always set it
        # even if data is effectively coalesced from SparseCooMatrix's perspective.
        internal_torch_sparse = internal_torch_sparse.coalesce()
        self.assertTrue(internal_torch_sparse.is_coalesced())
        self.assertLess(internal_torch_sparse.values().shape[0], values_non_coalesced.shape[0]) # nnz reduced

    def test_empty_coo_matrix(self):
        """Tests handling of empty SparseCooMatrix."""
        empty_coo = SparseCooMatrix(
            values=torch.empty(0, dtype=torch.float64),
            row_indices=torch.empty(0, dtype=torch.long),
            col_indices=torch.empty(0, dtype=torch.long),
            shape=(5,5)
        )
        self.assertEqual(empty_coo.values.numel(), 0)
        self.assertEqual(empty_coo.row_indices.numel(), 0)
        self.assertEqual(empty_coo.col_indices.numel(), 0)
        self.assertEqual(empty_coo.shape, (5,5))
        self.assertTrue(torch.equal(empty_coo.to_dense(), torch.zeros(5,5, dtype=torch.float64)))
        
        torch_sparse_empty = empty_coo.to_torch_sparse_coo()
        self.assertEqual(torch_sparse_empty.layout, torch.sparse_coo)
        self.assertEqual(torch_sparse_empty._nnz(), 0)
        self.assertEqual(tuple(torch_sparse_empty.shape), (5,5))
        self.assertTrue(torch.equal(torch_sparse_empty.to_dense(), torch.zeros(5,5, dtype=torch.float64)))

        # Test from_torch_sparse_coo with an empty sparse tensor
        empty_torch_coo = torch.sparse_coo_tensor(size=(3,3), dtype=torch.float64) # nnz = 0
        coo_from_empty_torch = SparseCooMatrix.from_torch_sparse_coo(empty_torch_coo)
        self.assertEqual(coo_from_empty_torch.values.numel(), 0)
        self.assertEqual(coo_from_empty_torch.shape, (3,3))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_coo_cuda(self):
        """Tests CUDA functionality for SparseCooMatrix."""
        v_cuda = self.v.cuda()
        r_cuda = self.r.cuda()
        c_cuda = self.c.cuda()
        expected_dense_cuda = self.expected_dense.cuda()
        
        coo_cuda = SparseCooMatrix(v_cuda, r_cuda, c_cuda, self.s)
        self.assertEqual(coo_cuda.values.device.type, 'cuda')
        self.assertEqual(coo_cuda.row_indices.device.type, 'cuda')
        self.assertEqual(coo_cuda.col_indices.device.type, 'cuda')

        dense_cuda = coo_cuda.to_dense()
        self.assertEqual(dense_cuda.device.type, 'cuda')
        self.assertTrue(torch.equal(dense_cuda, expected_dense_cuda))

        torch_sparse_cuda = coo_cuda.to_torch_sparse_coo()
        self.assertEqual(torch_sparse_cuda.device.type, 'cuda')
        self.assertTrue(torch.equal(torch_sparse_cuda.to_dense(), expected_dense_cuda))

        coo_from_torch_cuda = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_cuda)
        self.assertEqual(coo_from_torch_cuda.values.device.type, 'cuda')
        self.assertTrue(torch.equal(coo_from_torch_cuda.to_dense(), expected_dense_cuda))

    def test_coo_dtype(self):
        """Tests dtype handling for SparseCooMatrix."""
        v_f64 = self.v.to(torch.float64)
        expected_dense_f64 = self.expected_dense.to(torch.float64)

        coo_f64 = SparseCooMatrix(v_f64, self.r, self.c, self.s)
        self.assertEqual(coo_f64.values.dtype, torch.float64)
        self.assertTrue(torch.equal(coo_f64.to_dense(), expected_dense_f64))

        torch_sparse_f64 = coo_f64.to_torch_sparse_coo()
        self.assertEqual(torch_sparse_f64.dtype, torch.float64)
        self.assertTrue(torch.equal(torch_sparse_f64.to_dense(), expected_dense_f64))

        coo_from_torch_f64 = SparseCooMatrix.from_torch_sparse_coo(torch_sparse_f64)
        self.assertEqual(coo_from_torch_f64.values.dtype, torch.float64)
        self.assertTrue(torch.equal(coo_from_torch_f64.to_dense(), expected_dense_f64))

if __name__ == '__main__':
    unittest.main() 