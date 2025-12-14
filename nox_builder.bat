@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo      HA4DETR Windows Build - NOX Runner
echo ===============================================

:: -------------------------------
:: 1. Verify python exists
:: -------------------------------
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: python.exe not found in PATH.
    echo Install Python 3.11+ and retry.
    exit /b 1
)
echo Found vaild python. Checking that we have valid PIP as well.
:: -------------------------------
:: 2. Ensure pip works
:: -------------------------------
python -m pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip not available.
    echo Try reinstalling Python with "Add to PATH" enabled.
    exit /b 1
)
echo Found valid pip install, installaing nox.

:: -------------------------------
:: 3. Install nox if missing
:: -------------------------------
python -m pip show nox >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo nox not found. Installing nox...
    python -m pip install --upgrade nox
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed installing nox.
        exit /b 1
    )
)

echo ✔ nox is installed.
set DISTUTILS_USE_SDK=1
set MSSdk=1
:: -------------------------------
:: 4. Run NOX build session
::    Forward all args passed to this BAT
:: -------------------------------
echo Running: nox -s build-windows %*
nox -s build-windows %*
if %ERRORLEVEL% neq 0 (
    echo ERROR: nox build failed.
    exit /b 1
)
echo Successfully finish building with nox.

echo ===============================================
echo Build completed successfully!
echo Output wheels are under: dist\
echo ===============================================

endlocal
exit /b 0
