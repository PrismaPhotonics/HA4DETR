import os
import platform
import subprocess
import nox
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Default PyTorch spec
_DEFAULT_TORCH_SPEC = "torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121"


def _find_win_compiler_settings_file(version: str = "2019") -> Path:
    """
    Search for vcvars64.bat inside Visual Studio installation directory.
    Looks under Community, BuildTools, Enterprise, Professional.
    """
    assert platform.system() == "Windows"
    vs_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
                / "Microsoft Visual Studio" / version

    candidates = [
        "BuildTools",
        "Community",
        "Enterprise",
        "Professional",
    ]

    for edition in candidates:
        base = vs_root / edition / "VC" / "Auxiliary" / "Build"
        bat = base / "vcvars64.bat"
        if bat.exists():
            print(f"Found Visual Studio environment: {bat}")
            return bat

    raise FileNotFoundError(
        f"Could not find vcvars64.bat under '{vs_root}'. "
        "Please ensure Visual Studio Build Tools are installed."
    )

def _load_win_compiler(vs_version: str):
    """Load VS build environment into os.environ."""
    assert platform.system() == "Windows"
    print(f"→ Loading Visual Studio {vs_version} environment...")

    vcvars = _find_win_compiler_settings_file(version=vs_version)

    if not vcvars.exists():
        raise RuntimeError(f"vcvars64.bat not found: {vcvars}")
    # Run vcvars and capture environment
    cmd = f'"{vcvars}" && set'
    print(f"Running the commands from {vcvars} to set all env variables to the build")
    print(f"{cmd=}")
    result = subprocess.run(["cmd.exe", "/c", cmd],
                            capture_output=True, text=True)

    # if result.returncode != 0:
    #     raise RuntimeError("Failed running vcvars64.bat")

    # Import variables into this Python environment
    print("*"*40)
    print(f"The results STD:")
    print(F"{result.stdout}")
    print("*"*40)
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            print(f"setting {key} = {val}")
            os.environ[key] = val

    print("✔ Visual Studio environment loaded")


def _verify_cuda():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        raise RuntimeError("CUDA_HOME or CUDA_PATH is not set.")

    nvcc = Path(cuda_home) / "bin/nvcc.exe"
    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found at: {nvcc}")

    out = subprocess.check_output([str(nvcc), "--version"], text=True)
    print(out)

    if "release 12" not in out:
        raise RuntimeError("CUDA version 12.x required")

    print("✔ CUDA verified (12.x)")


@nox.session(name="build-wheel", python="3.11")
def build_wheel(session: nox.session):
    """
    Build GPU-enabled Windows wheel inside a nox-managed venv.
    """
    # Make sure to localte this at the root of the project!
    session.cd(ROOT)
    def _build_windows():

        session.log("===== ha4detr Windows Build =====")

        python = os.environ.get("PYTHON", "python")
        session.run(python, "-c", 'import platform; print("Python:", platform.python_version())')
        # ------------------------------
        # Load Visual Studio
        # ------------------------------
        vs_version = session.posargs[0] if session.posargs else "2019"
        session.log(f"running load builer for {vs_version} compiler version")
        _load_win_compiler(vs_version)
        os.environ["DISTUTILS_USE_SDK"] = "1"
        os.environ["MSSdk"] = "1"
        # ------------------------------
        # Validate CUDA
        # ------------------------------
        session.log("Verify that we have a valid version for CUDA to be used in this host")
        _verify_cuda()
        session.log("We have valid host to run this build on.")
        # ------------------------------
        # Install PyTorch + deps in venv
        # ------------------------------
        torch_spec = _DEFAULT_TORCH_SPEC
        if len(session.posargs) > 1:
            torch_spec = session.posargs[1]
        session.log(f"torch spect is {torch_spec}")
        session.install("numpy==2.0.0")
        session.install(*torch_spec.split())
        session.install("build", "wheel", "setuptools", "setuptools_scm", "numpy", "scipy", "pydantic")

        # ------------------------------
        # Build wheel
        # ------------------------------
        session.log("Running the build for this package")
        dist = ROOT / "dist"
        dist.mkdir(exist_ok=True)

        session.run("pip", "wheel", ".", "--no-build-isolation", "-w", "dist")

        # Validate wheel
        wheels = list(dist.glob("ha4detr*.whl"))
        if not wheels:
            session.error("Wheel build failed. No wheel found in dist/")
        session.log(f"✔ Build success: {wheels[0].name}")

    def _build_linux():
        """
        Build wheel on Linux.
        Assumes: CUDA 12.x installed, nvcc on PATH.
        """
        print("===== ha4detr Linux Build =====")

        # Ensure CUDA availability
        result = session.run("nvcc", "--version", log=False, success_codes=[0])
        print(result)

        session.install("build", "wheel", "setuptools", "setuptools_scm")
        session.install("numpy", "scipy", "pydantic")
        session.install(_DEFAULT_TORCH_SPEC.split()[0], *_DEFAULT_TORCH_SPEC.split()[1:])

        session.run("pip", "wheel", ".", "--no-build-isolation", "-w", "dist")

        wheels = list((ROOT / "dist").glob("ha4detr*.whl"))
        if not wheels:
            session.error("Wheel build failed on Linux")

    if platform.system() == "Windows":
        _build_windows()
    elif platform.system() == "Linux":
        _build_linux()
    else:
        raise SystemError("This cannot be run for system {platform.system()}")



@nox.session
def test(session: nox.Session):
    """Optional test session."""
    session.install(".")
    session.run("python", "-c", "import ha4detr; print('Loaded OK')")
