import os
import platform
import subprocess
import nox
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Default PyTorch spec
_DEFAULT_TORCH_SPEC = "torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121"



if platform.system() == "Windows":
    def _get_vcvars_path(target_version: str) -> str:
        vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    
        # Map friendly names to version ranges
        version_map = {
            "2017": "[15.0, 16.0)",
            "2019": "[16.0, 17.0)",
            "2022": "[17.0, 18.0)"
        }
        
        range_query = version_map.get(target_version)
        if not range_query:
            raise ValueError(f"Unsupported version request: {target_version}")

        cmd = [
            vswhere_path, 
            "-version", range_query,  # Filter for the specific version range
            "-products", "*", 
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        vs_install_path = result.stdout.strip()
        
        # If vswhere returns multiple paths (e.g. Community and Enterprise 2019), 
        # it returns them newline-separated. We'll take the first one.
        if not vs_install_path:
            raise RuntimeError(f"Visual Studio {target_version} with C++ tools was not found.")
        
        first_path = vs_install_path.splitlines()[0]
        bat_path = Path(first_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        
        return str(bat_path)

    def _load_msvc_env(vcvars_path: str):
        
        # Run the batch file and then run 'set' to see all variables
        query_cmd = f'"{vcvars_path}" && set'
        
        # Capture the output
        output = subprocess.check_output(query_cmd, shell=True, text=True)
        print(f"successsfully read {len(output)} vars for the env")
        # Parse the output and update Python's os.environ
        for line in output.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value


    def _load_win_compiler(version: str):
        vs_path = _get_vcvars_path(target_version=version)
        if not os.path.exists(vs_path):
            raise RuntimeError("Failed to find valid VS path in this host!")
        _load_msvc_env(vcvars_path=vs_path)


    
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

        print("CUDA verified (12.x)")

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
