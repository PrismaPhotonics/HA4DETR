"""
GPU Support
-----------
* Validated on CUDA 12.2 / driver ≥ 535 with a single NVIDIA RTX 4090 and
            on A5000 with  Driver Version: 580.95.05 CUDA Version: 13.0
* Requires PyTorch ≥ 2.1, SciPy ≥ 1.10, a C++17 tool-chain and NVCC

Run
---
$ python hungarian_gpu_batch.py        # builds extension, runs demo & check
"""

import torch

try:
    from . import _hungarian
except:
    from ha4detr import _hungarian as _hungarian


# ───────────────────────────── Python wrapper  ───────────────────────────────
def hungarian_gpu(cost: torch.Tensor, ncols: torch.Tensor) -> torch.Tensor:
    """
    Args:
        cost:  (B, _MAX_ROWS, _MAX_COLS)   float32 CUDA tensor
        ncols: (B,)                        int32 CUDA tensor
    Returns:
        out:   (B, _MAX_ROWS, 2)           int32 CUDA tensor, -1 padded
    Example:
        >>>import torch
        >>>from ha4detr import hungarian_batch
        >>>x = torch.rand(1,5,5, device='cuda')
        >>>n = torch.tensor([5], device='cuda').to(torch.int32)
        >>>print(hungarian_batch(x, n))
    """
    B, _, C = cost.shape
    out = torch.empty((B, C, 2), dtype=torch.int32, device=cost.device)
    _hungarian.hungarian_launcher(cost, ncols, out)
    return out  # already row-sorted
