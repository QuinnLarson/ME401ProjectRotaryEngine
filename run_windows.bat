@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================================================
echo   ME401 Rotary Hybrid Range-Extender Simulation - Windows Launcher
echo ========================================================================

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto check_deps

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -c "import numpy, scipy, matplotlib, pandas" >nul 2>&1
    if not errorlevel 1 (
        set "VENV_PY=..\.venv\Scripts\python.exe"
        goto run_sim
    )
)

echo [Setup] Searching for Python 3...
set "SYS_PY="

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3"
    goto create_venv
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=python"
    goto create_venv
)

python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=python3"
    goto create_venv
)

echo.
echo [ERROR] Python 3.10+ was not found on PATH.
echo Please install Python 3.11+ from https://www.python.org/downloads/
echo Make sure to check "Add python.exe to PATH" during installation.
pause
exit /b 1

:create_venv
echo [Setup] Creating virtual environment in .venv ...
%SYS_PY% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:check_deps
"%VENV_PY%" -c "import numpy, scipy, matplotlib, pandas" >nul 2>&1
if not errorlevel 1 goto run_sim

echo [Setup] Installing required packages from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed. Check internet connection and try again.
    pause
    exit /b 1
)

:run_sim
echo.
"%VENV_PY%" main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Simulation exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
