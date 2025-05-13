import torch

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""The primary device (CPU or CUDA GPU) for torch computations."""

DEFAULT_DTYPE = torch.float64
"""The default floating point precision for torch tensors (e.g., torch.float64, torch.float64)."""

torch.set_default_dtype(DEFAULT_DTYPE) 