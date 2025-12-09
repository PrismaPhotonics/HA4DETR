#pragma once
#include <torch/extension.h>

void hungarian_launcher(torch::Tensor cost, torch::Tensor ncols, torch::Tensor assignment);