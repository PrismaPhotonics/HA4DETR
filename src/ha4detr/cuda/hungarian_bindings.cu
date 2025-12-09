#include "hungarian_launcher.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


constexpr const int MAX_ROWS   = 300;       // workers (≥ tasks)
constexpr const int MAX_COLS   = 300;       // tasks, INF padded
constexpr const int WARP       = 32;
constexpr const float INF      = 1e20f;
constexpr const int _BLOCK_SIZE = 512;

/* ------------------------------- Utils ----------------------------------- */
template<int NT>
__device__ __forceinline__
void warp_min_reduce(float &val, int &idx)
{
    #pragma unroll
    for (int offset = NT / 2; offset > 0; offset >>= 1)
    {
        float val_other = __shfl_down_sync(0xffffffff, val , offset);
        int   idx_other = __shfl_down_sync(0xffffffff, idx , offset);
        if (val_other < val) {
            val = val_other;
            idx = idx_other;
        }
    }
}

/* ------------------------------------------------------------------------- *
 * One block solves one 300x300 assignment problem.
 * 512 threads cooperate; loops over rows are parallelised across threads,
 * while thread 0 performs the augmenting-path book-keeping.
 * We parallelise: ⭐ init          ⭐ Δ min-search     ⭐ u/v/minv update
 *                 ⭐ output write  ⭐ batch-grid over-subscription
 * ------------------------------------------------------------------------- */
template<int BLOCK_SIZE>
__global__ void hungarian_kernel(const float * __restrict__ cost,
                                 const int   * __restrict__ ncols,
                                 int         * __restrict__ assignment,
                                 int B,                         // batch size
                                 int cost_stride,               // 300 x 300
                                 int asgn_stride)               // 300 x 2
{
    int globalBid = blockIdx.x;
    const int tid = threadIdx.x;

    while (globalBid < B) {
        /* Pointers ---------------------------------------------------------------- */
        const float *costB = cost + globalBid * cost_stride;
        const int    cols  = ncols[globalBid];
        int         *asgnB = assignment + globalBid * asgn_stride;

        /* Shared Hungarian state -------------------------------------------------- */
        __shared__ float u[MAX_COLS + 1];               // task potentials
        __shared__ float v[MAX_ROWS + 1];               // worker potentials
        __shared__ int   p[MAX_ROWS + 1];               // matching: p[worker]=task
        __shared__ int   way[MAX_ROWS + 1];
        __shared__ float minv[MAX_ROWS + 1];
        __shared__ bool  used[MAX_ROWS + 1];
        __shared__ int   j0;
        __shared__ float delta_s;                       // best Δ in this iteration
        __shared__ int   j1_s;                          // argmin of Δ
        __shared__ bool  path_found;

        /* Init potentials (rows == cols == 300) ----------------------------------- */
        for (int k = tid; k <= MAX_ROWS; k += BLOCK_SIZE) {
            v[k] = u[k] = 0.0f;     // v[k] = 0.0f; if (k <= MAX_COLS) u[k] = 0.0f;
            p[k] = 0;
        }
        __syncthreads();

        /* Hungarian main loop over each column ------------------------------------ */
        for (int i = 1; i <= cols; ++i) {
            if (tid == 0) {
                p[0] = i;
                j0 = 0;
                path_found = false;
            }
            __syncthreads();

            /* Reset per-task buffers ---------------------------------------------- */
            for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) {
                minv[j] = INF;
                used[j] = false;
            }
            __syncthreads();

            /* grow alternating tree until an unmatched worker is found ------------ */
            while (!path_found) {
                /* Mark current row as used ---------------------------------------- */
                if (tid == 0) used[j0] = true;
                __syncthreads();

                /* Δ-scan: each thread processes stride-rows ----------------------- */
                int i0  = p[j0];                // current task index
                float best_val = INF;
                int best_j = 0;
                for (int j = tid + 1; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) continue;
                    float cur = costB[(j - 1) * MAX_COLS + (i0 - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < best_val) {
                        best_val = minv[j];
                        best_j = j;
                    }
                }

                /* Block-wide reduction to find global min Δ ----------------------- */
                const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP;
                warp_min_reduce<WARP>(best_val, best_j);
                if (BLOCK_SIZE > WARP) {
                    __shared__ float warp_min_val[WARPS_PER_BLOCK];
                    __shared__ int   warp_min_j[WARPS_PER_BLOCK];

                    if ((tid & (WARP - 1)) == 0) {
                        warp_min_val[tid / WARP] = best_val;
                        warp_min_j[tid / WARP] = best_j;
                    }
                    __syncthreads();

                    if (tid < WARPS_PER_BLOCK) {
                        best_val = warp_min_val[tid];
                        best_j = warp_min_j[tid];
                    }
                    else best_val = INF;

                    /* Final reduction --------------------------------------------- */
                    if (tid == 0) {
                        for (int k = 1; k < WARPS_PER_BLOCK; ++k)
                            if (warp_min_val[k] < best_val) {
                                best_val = warp_min_val[k];
                                best_j = warp_min_j[k];
                            }
                        delta_s = best_val;
                        j1_s = best_j;
                    }
                }
                else if (tid == 0) {
                    delta_s = best_val;
                    j1_s = best_j;
                }
                __syncthreads();

                /* Parallel update of u / v / minv --------------------------------- */
                for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) {
                        u[p[j]] += delta_s;
                        v[j] -= delta_s;
                    } else minv[j] -= delta_s;
                }
                __syncthreads();

                if (tid == 0) {
                    j0 = j1_s;
                    if (p[j0] == 0) path_found = true;
                }
                __syncthreads();
            }

            /* Augment along alternating path -------------------------------------- */
            if (tid == 0) {
                while (j0 != 0) {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                }
            }
            __syncthreads();
        }

        /* Output (row, col) pairs in parellel ------------------------------------- */
        for (int row = tid + 1; row <= MAX_ROWS; row += BLOCK_SIZE) {
            int task = p[row];
            asgnB[(row - 1) * 2] = row - 1;             // row index
            asgnB[(row - 1) * 2 + 1] = task - 1;        // col index
        }

        /* Move to next problem for oversubscription ------------------------------- */
        globalBid += gridDim.x;   // grid-stride over batch
        __syncthreads();
    }
}

/* ------------------------------- Launcher --------------------------------- */
void hungarian_launcher(torch::Tensor cost,
                        torch::Tensor ncols,
                        torch::Tensor assignment)
{
    TORCH_CHECK(cost.is_cuda(),                      "cost must be on CUDA");
    TORCH_CHECK(ncols.is_cuda(),                     "ncols must be on CUDA");
    TORCH_CHECK(assignment.is_cuda(),                "assignment must be on CUDA");
    TORCH_CHECK(cost.dtype() == torch::kFloat32,     "cost must be float32");
    TORCH_CHECK(ncols.dtype() == torch::kInt32,      "ncols must be int32");
    TORCH_CHECK(assignment.dtype() == torch::kInt32, "assignment must be int32");

    const int B = cost.size(0);
    const int R = cost.size(1);
    const int C = cost.size(2);

    const int cost_stride = R * C;
    const int asgn_stride = C * 2;

    TORCH_CHECK(ncols.size(0) == B, "ncols len must equal batch");

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    static constexpr const int threads = _BLOCK_SIZE;
    
    /* Oversubscribe: at least 4xSM block, no more than batch size (B) ------------- */
    int blocks  = min(B, smCount * 4);      // work when B > 512, RTX 4090: 128 SM

    hungarian_kernel<_BLOCK_SIZE><<<blocks, threads>>>(
        cost.data_ptr<float>(),
        ncols.data_ptr<int>(),
        assignment.data_ptr<int>(),
        B,
        cost_stride,
        asgn_stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}