@echo off
setlocal

REM --- Activate venv ---
call %~dp0build_env\Scripts\activate

REM --- Ensure environment for GPU build ---
set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"

echo Building wheel...
pip install wheel build
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip wheel . --no-build-isolation -w dist

echo.
echo DONE. Wheel files in dist/
dir dist

endlocal
pause
