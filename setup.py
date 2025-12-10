#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import platform
from pathlib import Path


_IS_WINDOWS_BUILD = platform.system() == "Windows"


def rel(path: Path) -> str:
    s = "./" + str(path.relative_to(ROOT)).replace("\\", "/")
    return s


if _IS_WINDOWS_BUILD:
    ROOT = Path(__file__).parent  # Path(__file__).parent.resolve()
else:
    ROOT = Path(__file__).parent

SRC = ROOT / "src"
SRC_BASE = SRC / "ha4detr"
SRC_CPU = SRC_BASE / "cpu"
SRC_CUDA = SRC_BASE / "cuda"


def get_extensions():
    from torch.utils.cpp_extension import (
        CUDAExtension,
        CppExtension,
        include_paths,
        library_paths,
        CUDA_HOME,
    )

    cpu_source_paths = [str(p) for p in SRC_CPU.glob("*.cpp")]
    cuda_source_paths = [str(p) for p in SRC_CUDA.glob("*.cu")]
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
        cuda_sources = [str(p.relative_to(ROOT)) for p in all_source_paths]
        include_dirs = include_paths()  # torch include directories
        library_dirs = library_paths(device_type="cuda")  # torch/lib directories
        rpath = "-Wl,-rpath,$ORIGIN/../torch/lib"
        extra_link_args = [rpath]
        extra_args = {
            "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
            "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
        }
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
        extra_args = {"cxx": ["/std:c++17"], "nvcc": ["-std=c++17"]}

        # Flatten torch include paths
        torch_includes = include_paths()
        # Our own include dir (absolute)
        local_include = str(SRC_BASE.resolve())

        include_dirs = [local_include] + torch_includes

        # Torch library dirs
        library_dirs = library_paths(device_type="cude")
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
