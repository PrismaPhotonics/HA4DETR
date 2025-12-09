#!/usr/bin/env python3
"""
GPU Support
-----------
* Validated on CUDA 12.2 / driver ≥ 535 with a single NVIDIA RTX 4090
* Requires PyTorch ≥ 2.1, SciPy ≥ 1.10, a C++17 tool-chain and NVCC

Run
---
$ python hungarian_gpu_batch.py        # builds extension, runs demo & check
"""

from . import _hungarian
import time
import torch

# from torch.utils.cpp_extension import load_inline
from scipy.optimize import linear_sum_assignment


# ───────────────────── Build / load inline extension ─────────────────────────
# os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # RTX 4090
# ext_mod = load_inline(
#     name="hungarian_gpu_batch_ext",
#     cpp_sources=[CPP_STUB],
#     cuda_sources=[CUDA_SRC],
#     functions=["hungarian_launcher"],
#     extra_cflags=["-std=c++17", "-O3"],
#     extra_cuda_cflags=["-std=c++17", "-O3", "--use_fast_math"],
#     verbose=False,
# )
# print(ext_mod.__file__)


# ───────────────────────────── Python wrapper  ───────────────────────────────
def hungarian_gpu(cost: torch.Tensor, ncols: torch.Tensor) -> torch.Tensor:
    """
    input:
        cost:  (B, _MAX_ROWS, _MAX_COLS)   float32 CUDA tensor
        ncols: (B,)                        int32 CUDA tensor
    return:
        out:   (B, _MAX_ROWS, 2)           int32 CUDA tensor, -1 padded
    """
    B, _, C = cost.shape
    out = torch.empty((B, C, 2), dtype=torch.int32, device=cost.device)
    _hungarian.hungarian_launcher(cost, ncols, out)
    return out  # already row-sorted


def timeit(fn, *args, sync=True, loops=1):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(loops):
        res = fn(*args)
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops, res


# ─────────────────────────────────── Demo (randomly padding) ────────────────────────────────────
# Function: hungarian_gpu
# - Input: A 3D float batched cost tensor of shape [B, _MAX_ROWS, _MAX_COLS],
#          where each [_MAX_ROWS, Ni] cost tensor (Ni ∈ [1, _MAX_COLS]) is padded
#          with inf to width _MAX_COLS before stacking.
#
#          A 1D int32 tensor of shape [B] indicating the real task numbers of cost tensors
#          (i.e., {Ni}, i ∈ [0, B - 1])
#
# - Output: A 3D integer tensor of shape [B, _MAX_ROWS, 2], where each (Ni, 2)
#           Hungarian assignment result (row i → column j) is padded with -1
#           to (_MAX_ROWS, 2) before stacking.
#
# - Performance Expectations: Maximize GPU parallelism to efficiently handle variable-sized
#                             cost matrices, achieving better speedup over
#                             scipy.optimize.linear_sum_assignment executed sequentially on the CPU.


def demo():
    assert torch.cuda.is_available()

    # Test settings
    device = "cuda"
    batch_size = 16
    n_rows, n_cols = 300, 300
    loops = 11

    gpu_t_total, cpu_t_total, speedup_total = 0, 0, 0
    for i in range(loops):

        # Generate the input cost tensor
        # (During DETR's training, the Hungarian algorithm inputs are stored on GPU; our demo follows this setup.)
        torch.manual_seed(i)
        Ns = torch.randint(1, 301, (batch_size,), device=device, dtype=torch.int32)
        cost = torch.full(
            (batch_size, n_rows, n_cols),
            float("inf"),
            device=device,
            dtype=torch.float32,
        )
        for b in range(cost.size(0)):
            Ni = Ns[b].item()
            cost[b, :, :Ni] = torch.rand(
                (n_rows, Ni), device=device, dtype=torch.float32
            )
        cost = cost.contiguous()

        # GPU implementation
        gpu_t, gpu_out = timeit(hungarian_gpu, cost, Ns)

        # CPU reference
        cpu_out = torch.full_like(gpu_out, int("-1"), device="cpu")
        cpu_out[:, :, 0] = torch.arange(n_cols).to(torch.int32)
        t0 = time.perf_counter()
        for b in range(cost.size(0)):
            Ni = Ns[b].item()
            r, c = linear_sum_assignment(cost[b][:, :Ni].cpu())
            cpu_out[b, r, 1] = torch.tensor(c).to(torch.int32)
        cpu_t = time.perf_counter() - t0

        # Checking: Check if the GPU and CPU results match.
        # (Due to non-uniqueness in the Hungarian algorithm, we consider results consistent if the final costs differ by less than 1e-4.)
        torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=0, atol=0)
        try:
            torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=0, atol=0)
        except AssertionError:
            indices_gpu = [
                (_gpu_out[:, 0][_valid[:, 1]], _gpu_out[:, 1][_valid[:, 1]])
                for _gpu_out, _valid in zip(gpu_out, gpu_out > -1)
            ]
            indices_cpu = [
                (_cpu_out[:, 0][_valid[:, 1]], _cpu_out[:, 1][_valid[:, 1]])
                for _cpu_out, _valid in zip(cpu_out, cpu_out > -1)
            ]
            err = 0
            for i in range(batch_size):
                c_gpu = (
                    cost[i][indices_gpu[i][0].cpu(), indices_gpu[i][1].cpu()]
                    .sum()
                    .item()
                )
                c_cpu = cost[i][indices_cpu[i][0], indices_cpu[i][1]].sum().item()
                err += abs(c_gpu - c_cpu)
            err /= batch_size
            if err < 1e-4:
                print(
                    f"WARNING: Alignments from CPU and GPU mismatch, but their real costs are equal [abs(ERROR) = {err}]."
                )
            else:
                raise AssertionError("Something wrong in GPU implemetation.")

        print(f"Mean valid ncols in cost matrix: {Ns.sum().item() / batch_size:5.2f}")
        print(f"GPU runtime      : {gpu_t*1e3:7.2f} ms")
        print(f"SciPy CPU runtime: {cpu_t*1e3:7.2f} ms")
        print(f"Speed-up         : {cpu_t / gpu_t:7.2f} x\n")

        # Skip Loop 0 due to GPU implementation warm-up
        gpu_t_total += gpu_t if i > 0 else 0
        cpu_t_total += cpu_t if i > 0 else 0
        speedup_total += cpu_t / gpu_t if i > 0 else 0
        time.sleep(1)

    if loops > 1:
        print(f"Average of the last {loops - 1} Loops")
        print(f"GPU runtime      : {(gpu_t_total / (loops - 1))*1e3:7.2f} ms")
        print(f"SciPy CPU runtime: {(cpu_t_total / (loops - 1))*1e3:7.2f} ms")
        print(f"Speed-up         : {(speedup_total / (loops - 1)):7.2f} x\n")


if __name__ == "__main__":
    demo()
