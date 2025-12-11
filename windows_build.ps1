<#
.SYNOPSIS
    Build the ha4detr Windows wheel.

.DESCRIPTION
    Validates Python, CUDA, Visual Studio, then builds a PyTorch CUDA extension wheel.

.PARAMETER PythonVersion
    Python version to use (default: 3.11).

.PARAMETER VSVersion
    Visual Studio version (2019 or 2022). Default: 2019.

.PARAMETER TorchSpec
    PyTorch package spec to install. Default:
    "torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121"

.PARAMETER Clean
    If set, deletes the virtual environment after build.

.EXAMPLE
    ./build_windows.ps1

.EXAMPLE
    ./build_windows.ps1 -Clean

.EXAMPLE
    ./build_windows.ps1 -PythonVersion 3.12 -TorchSpec "torch==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124"
#>

param(
    [string]$PythonVersion = "3.11",
    [ValidateSet("2019", "2022")]
    [string]$VSVersion = "2019",
    [string]$TorchSpec = "torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121",
    [switch]$Clean
)

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

Write-Host "========== ha4detr Windows Build Script ==========" -ForegroundColor Cyan
Write-Host "PythonVersion = $PythonVersion"
Write-Host "VSVersion     = $VSVersion"
Write-Host "TorchSpec     = $TorchSpec"
Write-Host "CleanEnv      = $($Clean.IsPresent)"
Write-Host "==================================================="

# -------------------------------
# 1. Verify python.exe exists
# -------------------------------
$python = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $python) { Fail "python.exe is not in PATH" }

$pyVersion = python - <<EOF
import platform, sys
print(platform.python_version())
EOF

if (-not ($pyVersion.StartsWith($PythonVersion))) {
    Fail "Python version mismatch. Expected $PythonVersion.x but found $pyVersion"
}
Write-Host "✔ Python version OK ($pyVersion)"

# -------------------------------
# 2. Check CUDA Toolkit
# -------------------------------
$cudaHome = $env:CUDA_HOME
if (-not $cudaHome) { $cudaHome = $env:CUDA_PATH }

if (-not $cudaHome) { Fail "CUDA_HOME is not set." }
if (-not (Test-Path "$cudaHome\bin\nvcc.exe")) {
    Fail "CUDA nvcc not found in $cudaHome"
}

$nvccVersion = & "$cudaHome\bin\nvcc.exe" --version
Write-Host "CUDA found at $cudaHome"
Write-Host $nvccVersion

if (-not ($nvccVersion -match "release 12")) {
    Fail "CUDA version < 12 detected. Please install CUDA 12.x"
}

Write-Host "✔ CUDA Toolkit OK"

# -------------------------------
# 3. Validate Visual Studio version
# -------------------------------
if ($VSVersion -eq "2019") {
    $vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019"
} else {
    $vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022"
}

if (-not (Test-Path $vsPath)) {
    Fail "Visual Studio $VSVersion not installed in expected path: $vsPath"
}
Write-Host "✔ Visual Studio $VSVersion detected"

# -------------------------------
# 4. Create virtual env
# -------------------------------
$VENV = ".venv_build"
if (Test-Path $VENV) { Remove-Item -Recurse -Force $VENV }

python -m venv $VENV
$venvActivate = Join-Path $VENV "Scripts\Activate.ps1"

. $venvActivate
Write-Host "✔ Virtual environment activated"

# -------------------------------
# 5. Install PyTorch
# -------------------------------
Write-Host "Installing PyTorch: $TorchSpec"
pip install $TorchSpec || Fail "Failed to install PyTorch"

# -------------------------------
# 6. Install build dependencies
# -------------------------------
pip install build wheel setuptools setuptools_scm numpy scipy pydantic || Fail "Failed installing deps"

# -------------------------------
# 7. Build the wheel
# -------------------------------
Write-Host "Building wheel..."
python -m build || Fail "Python build failed"

# -------------------------------
# 8. Verify wheel exists
# -------------------------------
$wheel = Get-ChildItem "dist\ha4detr*.whl" -ErrorAction SilentlyContinue
if (-not $wheel) { Fail "Wheel not found in dist/" }

Write-Host "✔ Build success: $($wheel.Name)"

# -------------------------------
# 9. Cleanup (optional)
# -------------------------------
if ($Clean.IsPresent) {
    Write-Host "Cleaning virtual environment..."
    Remove-Item -Recurse -Force $VENV
    Write-Host "✔ Environment cleaned."
}

Write-Host "==================================================="
Write-Host "Build completed successfully!"
Write-Host "Wheel located in: dist/"
Write-Host "==================================================="
