#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import platform
from pathlib import Path


_IS_WINDOWS_BUILD = platform.system() == "Windows"

ROOT = Path(__file__).parent

SRC = ROOT / "src"
SRC_BASE = SRC / "ha4detr"
SRC_CPU = SRC_BASE / "cpu"
SRC_CUDA = SRC_BASE / "cuda"


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)

def _verify_cuda_environment():
    """
    Ensures we only attempt to build on:
      - Windows or Linux
      - PyTorch with CUDA support
      - CUDA toolkit installed and visible to PyTorch
    """

    # --- Allowed platforms --------------------------------------------------
    system = platform.system()
    if system not in ("Windows", "Linux"):
        print(
            f"Unsupported platform '{system}' to build this on with CUDA support. "
            f"ha4detr only supports building on Windows or Linux with CUDA."
        )
        return True  # skip the reset we don't care.
    # --- Check PyTorch CUDA availability -----------------------------------
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        print("PyTorch must be installed BEFORE building ha4detr.")
        return False

    if not torch.cuda.is_available():
        print(
            "PyTorch reports that CUDA is NOT available on this system. "
            "Cannot build CUDA extension."
        )
        return False

    if not torch.backends.cuda.is_built():
        print(
            "Your PyTorch build was compiled WITHOUT CUDA. "
            "Install a CUDA-enabled PyTorch distribution."
        )
        return False

    torch_cuda = torch.version.cuda
    if not torch_cuda:
        print(
            "Torch does not expose a valid CUDA version (torch.version.cuda). "
            "Cannot safely build."
        )
        return False

    # --- Check system CUDA installation ------------------------------------
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_HOME

    if not cuda_home:
        print(
            "CUDA_HOME / CUDA_PATH is not set. "
            "Please set it to your local CUDA toolkit path."
        )
        return False
    exe_postfix = "nvcc.exe" if _IS_WINDOWS_BUILD else "nvcc"
    nvcc = Path(cuda_home) / "bin" / exe_postfix
    if not nvcc.exists():
        print(
            f"nvcc not found at expected path: {nvcc}"
        )
        print("Your CUDA installation is incomplete or incorrect.")
        return False

    # --- Optional: check major version compatibility -----------------------
    # torch.version.cuda is like "12.1" or "12.6"
    torch_major = torch_cuda.split(".")[0]
    try:
        system_nvcc_output = os.popen(f'"{nvcc}" --version').read().lower()
    except Exception:
        system_nvcc_output = ""

    if torch_major not in system_nvcc_output:
        print("*************************************************************")
        print("* WARNING: CUDA version mismatch between PyTorch and system *")
        print("* This may still work, but is not guaranteed.               *")
        print("* Torch CUDA:", torch_cuda)
        print("* nvcc --version output:", system_nvcc_output)
        print("*************************************************************")

    print("CUDA environment OK")
    return True


def get_extensions():
    from torch.utils.cpp_extension import (
        CUDAExtension,
        CppExtension,
        include_paths,
        library_paths,
        CUDA_HOME,
    )

    assert _verify_cuda_environment(), "You cannot run this build on this system - you would need to update it to match requirement!"

    cpu_source_paths = [rel(p) for p in SRC_CPU.glob("*.cpp")]
    cuda_source_paths = [rel(p) for p in SRC_CUDA.glob("*.cu")]
    all_source_paths = cpu_source_paths + cuda_source_paths

    def _debug_print():
        from torch import version

        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        print("*" * 40)
        print(f"{CUDA_HOME=}, {cuda_home=}")
        print(f"{os.getenv('CUDA_HOME')=}")
        print(f"{version.cuda=}")
        print("*" * 40)

    def _get_linux_settings():
        local_includes = str(SRC_BASE)
        # 2. Collect CUDA sources (from the new 'cuda' subdirectory)
        # str(Path(p).relative_to(ROOT))
        cuda_sources = [p for p in all_source_paths]
        include_dirs = include_paths()  # torch include directories
        library_dirs = library_paths(device_type="cuda")  # torch/lib directories
        rpath = "-Wl,-rpath,$ORIGIN/../torch/lib"
        extra_link_args = [rpath]
        extra_args = {
            "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
            "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
        }
        print("="*40)
        print(f"{cuda_sources=}")
        print("="*40)
        return [
            CUDAExtension(
                name="ha4detr._hungarian",
                sources=cuda_sources,
                include_dirs=[local_includes, include_dirs],
                library_dirs=library_dirs,
                extra_link_args=extra_link_args,
                extra_compile_args=extra_args,
            )
        ]

    def _get_windows_settings():
        extra_args = {
            "cxx": ["/std:c++17", "/Zc:__cplusplus"],
            "nvcc": [
                "-std=c++17",
                "-Xcompiler=/std:c++17",
                "-Xcompiler=/Zc:__cplusplus",
            ],
        }

        # Flatten torch include paths
        torch_includes = include_paths()
        # Our own include dir (absolute)
        local_include = str(SRC_BASE.resolve())

        include_dirs = [local_include] + torch_includes

        # Torch library dirs
        library_dirs = library_paths(cuda=True)
        print("#" * 40)
        print(f"my sources for the build are {all_source_paths}")
        return [
            CUDAExtension(
                name="ha4detr._hungarian",
                sources=all_source_paths,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_compile_args=extra_args,
            )
        ]

    system = platform.system()
    _debug_print()

    if _IS_WINDOWS_BUILD:
        print("=" * 40)
        print(" building for WINDOWS")
        print("=" * 40)
        return _get_windows_settings()
    if system == "Linux":
        return _get_linux_settings()
    if system == "Darwin":
        # CPU fallback for macOS
        return [
            CppExtension(
                name="ha4detr._hungarian",
                include_dirs=[str(SRC_BASE)],
                sources=[str(p.relative_to(ROOT)) for p in cpu_source_paths],
                extra_compile_args={
                    "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
                    "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
                },
            )
        ]

    return []


def build_ext():
    # Import BuildExtension lazily for same reason as above
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension


setup(
    name="ha4detr",
    packages=find_packages(where="src", exclude=["ha4detr.cpu", "ha4detr.cuda"]),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext()},
)
