import torch
import unittest
from torchls.sparse import SparseCsrMatrix # Adjusted import path

class TestSparseCsrMatrix(unittest.TestCase):

    def setUp(self):
        """Set up common data for tests."""
        self.v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        self.ci = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long) # col_indices
        self.cr = torch.tensor([0, 2, 3, 5], dtype=torch.long)    # crow_indices
        self.s = (3, 4) # num_rows = 3, num_cols = 4
        self.expected_dense = torch.tensor([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 5.0, 0.0]
        ], dtype=torch.float64)

        # Data for creating from torch sparse tensor
        v_alt = torch.tensor([10., 20., 30., 40., 50.], dtype=torch.float64)
        ci_alt = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
        cr_alt = torch.tensor([0, 2, 3, 5], dtype=torch.long)
        s_alt = (3, 4)
        self.torch_sparse_tensor_alt = torch.sparse_csr_tensor(cr_alt, ci_alt, v_alt, s_alt)
        self.expected_dense_alt = torch.tensor([
            [10.0, 0.0, 20.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
            [40.0, 0.0, 50.0, 0.0]
        ], dtype=torch.float64)

    def test_csr_matrix_creation(self):
        """Tests basic creation of SparseCsrMatrix."""
        csr_matrix = SparseCsrMatrix(
            values=self.v,
            col_indices=self.ci,
            crow_indices=self.cr,
            shape=self.s
        )
        self.assertTrue(torch.equal(csr_matrix.values, self.v))
        self.assertTrue(torch.equal(csr_matrix.col_indices, self.ci))
        self.assertTrue(torch.equal(csr_matrix.crow_indices, self.cr))
        self.assertEqual(csr_matrix.shape, self.s)
        # print(f"Created SparseCsrMatrix: {csr_matrix}") # Optional print

    def test_csr_to_dense(self):
        """Tests conversion of SparseCsrMatrix to a dense tensor."""
        csr_matrix = SparseCsrMatrix(
            values=self.v,
            col_indices=self.ci,
            crow_indices=self.cr,
            shape=self.s
        )
        dense_matrix = csr_matrix.to_dense()
        self.assertTrue(torch.equal(dense_matrix, self.expected_dense))
        # print(f"Dense matrix from CSR: \n{dense_matrix}") # Optional print

    def test_csr_to_torch_sparse_csr(self):
        """Tests conversion to PyTorch's sparse CSR tensor."""
        csr_matrix = SparseCsrMatrix(
            values=self.v,
            col_indices=self.ci,
            crow_indices=self.cr,
            shape=self.s
        )
        torch_sparse_csr = csr_matrix.to_torch_sparse_csr()
        self.assertTrue(torch_sparse_csr.is_sparse_csr)
        self.assertTrue(torch.equal(torch_sparse_csr.to_dense(), self.expected_dense))
        self.assertTrue(torch.equal(torch_sparse_csr.values(), self.v))
        self.assertTrue(torch.equal(torch_sparse_csr.col_indices(), self.ci))
        self.assertTrue(torch.equal(torch_sparse_csr.crow_indices(), self.cr))
        self.assertEqual(tuple(torch_sparse_csr.shape), self.s)
        # print(f"PyTorch sparse CSR tensor: \n{torch_sparse_csr}") # Optional print

    def test_csr_from_torch_sparse_csr(self):
        """Tests creation from PyTorch's sparse CSR tensor."""
        torch_sparse_tensor = self.torch_sparse_tensor_alt
        csr_from_torch = SparseCsrMatrix.from_torch_sparse_csr(torch_sparse_tensor)
        
        self.assertTrue(torch.equal(csr_from_torch.values, torch_sparse_tensor.values()))
        self.assertTrue(torch.equal(csr_from_torch.col_indices, torch_sparse_tensor.col_indices()))
        self.assertTrue(torch.equal(csr_from_torch.crow_indices, torch_sparse_tensor.crow_indices()))
        self.assertEqual(csr_from_torch.shape, tuple(torch_sparse_tensor.shape))
        self.assertTrue(torch.equal(csr_from_torch.to_dense(), self.expected_dense_alt))
        # print(f"SparseCsrMatrix from torch sparse: {csr_from_torch}") # Optional print

    def test_csr_validation_errors(self):
        """Tests that appropriate ValueErrors are raised for invalid inputs."""
        v, ci, cr, s = self.v, self.ci, self.cr, self.s

        with self.assertRaisesRegex(ValueError, "must be 1D tensors"):
            SparseCsrMatrix(v, ci, torch.tensor([[0],[1]]), s) # crow_indices not 1D
        with self.assertRaisesRegex(ValueError, "values and col_indices must have the same length"):
            SparseCsrMatrix(v[:-1], ci, cr, s)
        with self.assertRaisesRegex(ValueError, "Shape must be a 2-tuple"):
            SparseCsrMatrix(v, ci, cr, (3,))
        with self.assertRaisesRegex(ValueError, "crow_indices must have length num_rows \\+ 1"):
            SparseCsrMatrix(v, ci, cr[:-1], s) # Incorrect crow_indices length
        with self.assertRaisesRegex(ValueError, "Column indices are out of bounds"):
            SparseCsrMatrix(v, torch.tensor([0, 2, 1, 0, 4]), cr, s) # Col index 4 out of bounds for shape (3,4)
        with self.assertRaisesRegex(ValueError, "First element of crow_indices must be 0"):
            SparseCsrMatrix(v, ci, torch.tensor([1, 2, 3, 5]), s)
        with self.assertRaisesRegex(ValueError, "Last element of crow_indices must be nnz"):
            SparseCsrMatrix(v, ci, torch.tensor([0, 2, 3, 4]), s) # nnz is 5
        with self.assertRaisesRegex(ValueError, "crow_indices must be monotonically non-decreasing"):
            SparseCsrMatrix(v, ci, torch.tensor([0, 3, 2, 5]), s)

    def test_empty_csr_matrix(self):
        """Tests handling of empty SparseCsrMatrix."""
        # 0x0 matrix
        empty_csr_0x0 = SparseCsrMatrix(
            values=torch.empty(0, dtype=torch.float64),
            col_indices=torch.empty(0, dtype=torch.long),
            crow_indices=torch.tensor([0], dtype=torch.long), # num_rows=0 -> len=1
            shape=(0,0)
        )
        self.assertEqual(empty_csr_0x0.values.numel(), 0)
        self.assertEqual(empty_csr_0x0.shape, (0,0))
        dense_0x0 = empty_csr_0x0.to_dense()
        self.assertEqual(dense_0x0.shape, (0,0))
        self.assertEqual(dense_0x0.dtype, torch.float64)

        torch_sparse_0x0 = empty_csr_0x0.to_torch_sparse_csr()
        self.assertTrue(torch_sparse_0x0.is_sparse_csr)
        self.assertEqual(torch_sparse_0x0._nnz(), 0)
        self.assertEqual(tuple(torch_sparse_0x0.shape), (0,0))
        self.assertEqual(torch_sparse_0x0.to_dense().shape, (0,0))

        # Matrix with rows/cols but 0 nnz
        empty_csr_3x4_nnz0 = SparseCsrMatrix(
            values=torch.empty(0, dtype=torch.float64),
            col_indices=torch.empty(0, dtype=torch.long),
            crow_indices=torch.tensor([0,0,0,0], dtype=torch.long), # num_rows=3 -> len=4
            shape=(3,4)
        )
        self.assertEqual(empty_csr_3x4_nnz0.values.numel(), 0)
        self.assertEqual(empty_csr_3x4_nnz0.shape, (3,4))
        self.assertTrue(torch.equal(empty_csr_3x4_nnz0.to_dense(), torch.zeros(3,4, dtype=torch.float64)))

        torch_sparse_empty_nnz0 = empty_csr_3x4_nnz0.to_torch_sparse_csr()
        self.assertEqual(torch_sparse_empty_nnz0._nnz(), 0)
        self.assertEqual(tuple(torch_sparse_empty_nnz0.shape), (3,4))
        self.assertTrue(torch.equal(torch_sparse_empty_nnz0.to_dense(), torch.zeros(3,4, dtype=torch.float64)))

        # Test from_torch_sparse_csr with an empty sparse tensor
        empty_torch_csr = torch.sparse_csr_tensor(
            torch.tensor([0,0,0], dtype=torch.long), # crow_indices for 2 rows
            torch.empty((0), dtype=torch.long),      # col_indices
            torch.empty((0), dtype=torch.float64),   # values
            size=(2,3)
        )
        csr_from_empty_torch = SparseCsrMatrix.from_torch_sparse_csr(empty_torch_csr)
        self.assertEqual(csr_from_empty_torch.values.numel(), 0)
        self.assertEqual(csr_from_empty_torch.shape, (2,3))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_csr_cuda(self):
        """Tests CUDA functionality for SparseCsrMatrix."""
        v_cuda = self.v.cuda()
        ci_cuda = self.ci.cuda()
        cr_cuda = self.cr.cuda()
        expected_dense_cuda = self.expected_dense.cuda()
        
        csr_cuda = SparseCsrMatrix(v_cuda, ci_cuda, cr_cuda, self.s)
        self.assertEqual(csr_cuda.values.device.type, 'cuda')
        self.assertEqual(csr_cuda.col_indices.device.type, 'cuda')
        self.assertEqual(csr_cuda.crow_indices.device.type, 'cuda')

        dense_cuda = csr_cuda.to_dense()
        self.assertEqual(dense_cuda.device.type, 'cuda')
        self.assertTrue(torch.equal(dense_cuda, expected_dense_cuda))

        torch_sparse_cuda = csr_cuda.to_torch_sparse_csr()
        self.assertEqual(torch_sparse_cuda.device.type, 'cuda')
        self.assertTrue(torch.equal(torch_sparse_cuda.to_dense(), expected_dense_cuda))

        csr_from_torch_cuda = SparseCsrMatrix.from_torch_sparse_csr(torch_sparse_cuda)
        self.assertEqual(csr_from_torch_cuda.values.device.type, 'cuda')
        self.assertTrue(torch.equal(csr_from_torch_cuda.to_dense(), expected_dense_cuda))

    def test_csr_dtype(self):
        """Tests dtype handling for SparseCsrMatrix."""
        v_f64 = self.v.to(torch.float64)
        expected_dense_f64 = self.expected_dense.to(torch.float64)

        csr_f64 = SparseCsrMatrix(v_f64, self.ci, self.cr, self.s)
        self.assertEqual(csr_f64.values.dtype, torch.float64)
        self.assertTrue(torch.equal(csr_f64.to_dense(), expected_dense_f64))

        torch_sparse_f64 = csr_f64.to_torch_sparse_csr()
        self.assertEqual(torch_sparse_f64.dtype, torch.float64) # This will be values.dtype
        self.assertTrue(torch.equal(torch_sparse_f64.to_dense(), expected_dense_f64))

        csr_from_torch_f64 = SparseCsrMatrix.from_torch_sparse_csr(torch_sparse_f64)
        self.assertEqual(csr_from_torch_f64.values.dtype, torch.float64)
        self.assertTrue(torch.equal(csr_from_torch_f64.to_dense(), expected_dense_f64))

if __name__ == '__main__':
    unittest.main() 