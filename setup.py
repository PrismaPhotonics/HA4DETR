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
    extra_link_args = [f"-Wl,-rpath,{p}" for p in library_dirs]

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

# import os
# from setuptools import setup
# import glob
# import platform
# from pathlib import Path

# from setuptools import find_packages
# from setuptools import setup

# #requirements = ["torch", "torchvision"]

# SRC_DIR = Path("src") / "ha4detr"


# def get_extensions():
#     from torch.utils.cpp_extension import CUDAExtension, CppExtension, CUDA_HOME

#     system = platform.system()
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     extensions_dir = os.path.join(this_dir, "src")
#     main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
#     source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
#     source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

#     sources = main_file + source_cpu
#     extension = CppExtension
#     extra_compile_args = {"cxx": []}
#     define_macros = []

#     # This is your main implementation; adjust filenames if you add a .cpp
#     # cuda_sources = [
#     #     str(SRC_DIR / "hungarian_launcher.cu"),
#     #     str(SRC_DIR / "hungarian_bindings.cpp"),
#     # ]
#     # If you later add C++ bindings, add them here:
#     # common_sources = [str(SRC_DIR / "hungarian_bindings.cpp")]

#     if torch.cuda.is_available() and CUDA_HOME is not None:
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]
#         # CUDA build (requires NVCC + CUDA toolkit)
#         # return [
#         #     CUDAExtension(
#         #         name="ha4detr._hungarian",
#         #         sources=cuda_sources,
#         #     )
#         # ]
#         sources = [os.path.join(extensions_dir, s) for s in sources]
#         include_dirs = [extensions_dir]
#         ext_modules = [
#             extension(
#                 "MultiScaleDeformableAttention",
#                 sources,
#                 include_dirs=include_dirs,
#                 define_macros=d,
#             )
#         ]
#         return ext_modules
#     elif system in {"Darwin"}:
#         # CPU-only fallback – no CUDA on macOS
#         return [
#             CppExtension(
#                 name="ha4detr._hungarian",
#                 sources=[str(SRC_DIR / "hungarian_bindings.cpp")],
#             )
#         ]
#     else:
#         return []


# def build_ext():
#     # Import BuildExtension lazily for same reason as above
#     from torch.utils.cpp_extension import BuildExtension

#     return BuildExtension


# setup(
#     ext_modules=get_extensions(),
#     cmdclass={"build_ext": build_ext()},
# )
