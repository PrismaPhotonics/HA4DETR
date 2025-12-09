#include "hungarian_launcher.h"
#include <torch/extension.h>

// Declare the CUDA kernel (implemented in hungarian_launcher.cu)
void hungarian_launcher(torch::Tensor cost,
                        torch::Tensor ncols,
                        torch::Tensor assignment);

// Python-visible wrapper with proper checks
void hungarian_launcher_wrapper(torch::Tensor cost,
                                torch::Tensor ncols,
                                torch::Tensor assignment) {
    TORCH_CHECK(cost.is_cuda(), "cost tensor must be CUDA");
    TORCH_CHECK(ncols.is_cuda(), "ncols tensor must be CUDA");
    TORCH_CHECK(assignment.is_cuda(), "assignment tensor must be CUDA");
    hungarian_launcher(cost, ncols, assignment);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "hungarian_launcher",
        &hungarian_launcher_wrapper,
        "Hungarian CUDA kernel"
    );
}
