# GPU-Accelerated Batched Hungarian Algorithm for DETR

<p align="center">
<img align="center" src="https://github.com/linfeng93/HA4DETR/blob/main/logo.jpg" width="width:200" height="200" alt="Intellifusion Logo">
<p>
  
English | [中文](./README_zh.md)
  
This repository provides an efficient **CUDA/C++ extension** of the Hungarian Algorithm, seamlessly integrated with PyTorch and optimized for batched 300 × N (N ≤ 300) assignment problems—commonly encountered in the training of **DETR** and related models.

Compared to the widely-used SciPy's CPU-based `linear_sum_assignment`, this implementation delivers a **8~160× speed-up** on random inputs and a **3~10× speed-up** in practical DETR training scenarios, evaluated on **NVIDIA RTX 4090 GPUs**.

---

## 🚀 Features

- Supports batched assignment problems of shape [B × 300 × N]
  - The maximum size of each cost matrix can be adjusted by updating `MAX_ROWS` and `MAX_COLS` in the CUDA source code before compiling
  - Each cost matrix can have a different size; pad to size N with INF values before stacking (N ≤ 300)
  - Uses `torch.float32` as the data type for cost matrices

- GPU parallelization of the Hungarian Algorithm includes:
  - Initialization of dual variables
  - Δ-minimum search with warp-level reduction
  - Potentials update (u, v, minv)
  - Final assignment and write-out

- Fully validated against SciPy's `linear_sum_assignment` in terms of:
  - Correctness
  - Runtime performance

---

## 📦 Requirements

- Python 3.x
- PyTorch ≥ 2.0 (with `torch.utils.cpp_extension`)
- NVIDIA driver ≥ 535
- CUDA ≥ 12.2 (earlier versions may work)
- C++17-compatible compiler and NVCC
  
> Tested on: NVIDIA RTX 4090 GPU

---
## Building
You can use the existing docker file here to build a Linux version to without the need to install all of nvidia dependencies as well as torch.
This dockerfile build for Linux only as docker with GPU is only supported for such OS.
>Note that with MacOS it would not work regardless as it lack CUDA support.
>Make sure that you have GPU on you system:
```cmd
nvidia-smi
```

### Linux Build
To run the build with the docker file:
1. First build the docker image:
```bash
docker build --network host -t ha4detr-builder -f Dockerfile.builder .
```
2. Create the package:
> Note that we are using the sources from this directory into the docker image, and assuming you're building from the git repo current directory:
```bash
docker run --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ha4detr-build \
  bash -c "python -m build"
```
However it is better to let the script `build_with_docker.sh` do the work for you.
The script will ensure that:
- You have the docker image ready.
- Clean old version of the package.
- build the package inside the docker image.
- Rename the package based on the current git tag.

### Windows Build
Note:
- You would need to have Nvidia CUDA tool kit version 12.1 or 12.0 to build on Windows.
- You would need to have Visual Studio version 2022 or newer to build.
To build this with `windows` you would need to:
- Install the `Nvidia` CUDA development toolkit from [Nvidia web site](archive).
- Have a building environment for c++ builds - i.e. Visual Studio. Follow this [link](https://aka.ms/vs/17/release/vs_BuildTools.exe), and then [this getting started](https://visualstudio.microsoft.com/vs/getting-started/).
- Install [Pyenv](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#installation), and setup `python 3.11` as the global default python version.
- Use the `Developer Command Prompt for VS 20<version>`
The MSVC environment must be pre-activated.
>You MUST NOT run the build from plain CMD or PowerShell.
Steps:
  Click Start
  Search:
    "x64 Native Tools Command Prompt for VS 2022"
    (or "Developer Command Prompt for VS 2022")
  Open it
- Setup python virtual environment with:
```cmd
python -m venv <my venv name>
<my venv name>\Scripts\activate
```
- Install pytoch with CUDA support:
```cmd
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```
- Install build dependencies:
```cmd
pip install wheel build
```

- Make sure that you have the following env variables sets:
> CUDA_HOME - should point to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`
> The nvcc is working: 
```cmd
nvcc --version
```
- setup the required environment variables:
```cmd
set DISTUTILS_USE_SDK=1
set MSSdk=1
```
- Build the package with:

```cmd
pip wheel . --no-build-isolation -w dist
```

#### Build Automation
While we cannot build on windows machine with fully isolated environment like we can using dockers on Linux, we have a build script for windows that can automated some of the build environment for us.
To execute this script with all default:
```cmd
./scripts/build_windows.ps1
```
The script has these options and defaults:
- Python version (≥3.11)
- CUDA toolkit version (≥12.0, and matches Torch)
- Visual Studio toolchain version (default VS2019)
- PyTorch version (default 2.2.0+cu121)
- Creates a temporary virtual environment and cleans it afterwards
- Produces a final wheel inside dist/
- Provides command-line options to customize behavior
Note that we assuming that you have the correct python version - you can use `pyenv` and set it to `python 3.11`, but you do need to install VS version 2019 or 2022 and you do need to install CUDA toolkit 12 to 12.3 for this to work.
##### Examples:
- Running with custom `pytorch`
```cmd
./scripts/build_windows.ps1 -TorchSpec "torch==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124"
```
- Running with `python 3.12`
```cmd
./scripts/build_windows.ps1 -PythonVersion 3.12
```
- Doing virtual env cleanup at the end:
```cmd
./scripts/build_windows.ps1 -Clean
```
> Note that you can combine all of the above.



## ⚙️ How to Use

**1. Run the demo script**

```bash
python hungarian_gpu_batch.py
```

This will:
- Compile the CUDA/C++ extension inline
- Run tests using random cost matrices
- Compare GPU results with SciPy CPU results
- Report runtime speed-up statistics

You will see per-trial performance logs followed by average statistics across the last 10 runs. Example output:
```log
Mean valid ncols in cost matrix: 203.75
GPU runtime      :    3.85 ms
SciPy CPU runtime:   32.69 ms
Speed-up         :    8.50 x

Mean valid ncols in cost matrix: 164.62
GPU runtime      :    3.23 ms
SciPy CPU runtime:   29.52 ms
Speed-up         :    9.13 x

...
...

Average of the last 10 Loops
GPU runtime      :    1.49 ms
SciPy CPU runtime:   24.67 ms
Speed-up         :   20.67 x
```

**2. Use in your own code**

A simple example:

```python
import torch
from hungarian_gpu_batch import hungarian_gpu

cost = torch.rand((16, 300, 300), device="cuda", dtype=torch.float32)  # Batched cost matrices
Ns = torch.randint(1, 301, (16,), device="cuda", dtype=torch.int32)    # Batched actual task numbers, each Ni corresponds to one cost matrix (entries beyond Ni are ignored)

output = hungarian_gpu(cost, Ns)
```

---

## 📊 Performance

**1. Random Testing**

We present a speed-up curve with respect to varying batch sizes (B), using randomly padded input cost matrices of shape [B × 300 × 300]. Experiments are conducted on a single NVIDIA RTX 4090 GPU.

> Baseline: SciPy's `linear_sum_assignment`

<img src="https://github.com/linfeng93/HA4DETR/blob/main/speedup.png" style="width:70%; height:auto;">

**2. Real-World Application**

We integrate this GPU-based Hungarian Algorithm into the DETR training pipeline by replacing SciPy's `linear_sum_assignment`.

With a per-GPU batch size of 16 (each GPU processing its own local samples), this implementation achieves a **3× speed-up**, leading to an overall **10%** reduction in training overhead. As batch size increases, the speed-up would become even more pronounced, consistent with the trends observed in random testing.

---

## 📜 License

This repository is licensed under the [Apache-2.0 license](https://github.com/linfeng93/HA4DETR/blob/main/LICENSE).

---

## 📚 Citation
If you find this repository useful, please cite it using the following BibTeX:
```bibtex
@misc{ha4detr,
    title = {GPU-Accelerated Batched Hungarian Algorithm for DETR},
    author = {Feng Lin, Xiaotian Yu, Rong Xiao},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/linfeng93/HA4DETR},
}
```

---

## 💼 Acknowledgements

This open-source repository originates from a video object detection project at **Intellifusion Inc.**, and is developed by Feng Lin, Xiaotian Yu, and Rong Xiao.

---

## 📮 Contact

Feel free to open an issue or PR for discussions, improvements, or questions.
