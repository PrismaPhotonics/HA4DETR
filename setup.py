#!/usr/bin/env python

from setuptools import setup, find_packages
import platform
from pathlib import Path

ROOT = Path(__file__).parent


def rel(path: Path) -> str:
    s = "./" + str(path.relative_to(ROOT)).replace("\\", "/")
    return s


SRC = ROOT / "src"
SRC_BASE = SRC / "ha4detr"
SRC_CPU = SRC_BASE / "cpu"
SRC_CUDA = SRC_BASE / "cuda"
INCLUDE_PATH_BASE = INCLUDE_PATH_BASE = str(SRC_BASE)


def get_extensions():
    from torch.utils.cpp_extension import (
        CUDAExtension,
        CppExtension,
        include_paths,
        library_paths,
    )

    system = platform.system()
    cpu_source_paths = list(SRC_CPU.glob("*.cpp"))

    # 2. Collect CUDA sources (from the new 'cuda' subdirectory)
    cuda_source_paths = list(SRC_CUDA.glob("*.cu"))
    all_source_paths = cpu_source_paths + cuda_source_paths
    cuda_sources = [str(p.relative_to(ROOT)) for p in all_source_paths]
    include_dirs = include_paths()  # torch include directories
    library_dirs = library_paths()  # torch/lib directories
    rpath = "-Wl,-rpath,$ORIGIN/../torch/lib"
    extra_link_args = [rpath]

    if system in {"Linux", "Windows"}:
        # Build CUDA extension on Linux and Windows
        return [
            CUDAExtension(
                name="ha4detr._hungarian",
                sources=cuda_sources,
                include_dirs=[INCLUDE_PATH_BASE, include_dirs],
                library_dirs=library_dirs,
                extra_link_args=extra_link_args,
                extra_compile_args={
                    "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
                    "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"],
                },
            )
        ]
    if system == "Darwin":
        # CPU fallback for macOS
        return [
            CppExtension(
                name="ha4detr._hungarian",
                include_dirs=[INCLUDE_PATH_BASE],
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
    packages=find_packages(where="src", exclude=["ha4detr.cpu", "ha4detr.cuda"]),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext()},
)
