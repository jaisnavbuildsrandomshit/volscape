@echo off
setlocal EnableExtensions
rem VOLSCAPE launcher (Windows). Creates .venv next to this file, installs pinned deps, starts the app.
cd /d "%~dp0"

set "PY="
for %%V in (3.12 3.13 3.11 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [VOLSCAPE] ERROR: Python 3.11+ was not found.
  echo            Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and re-run start.cmd.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [VOLSCAPE] Creating virtual environment with %PY% ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [VOLSCAPE] ERROR: could not create .venv. Check disk permissions, or delete a partial .venv folder and retry.
    exit /b 1
  )
)

set "VPY=.venv\Scripts\python.exe"
fc /b requirements.txt .venv\.installed-requirements >nul 2>&1
if errorlevel 1 (
  echo [VOLSCAPE] Installing pinned dependencies ^(first run only, needs internet once^) ...
  "%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 (
    echo [VOLSCAPE] ERROR: dependency installation failed. Check your internet connection or proxy, then re-run.
    echo            Python 3.11-3.13 is recommended; very new Python versions may lack prebuilt wheels.
    exit /b 1
  )
  copy /y requirements.txt .venv\.installed-requirements >nul
)

if /i "%~1"=="test" (
  "%VPY%" -m pytest -q
  exit /b %errorlevel%
)

echo [VOLSCAPE] Starting at http://127.0.0.1:8501  ^(Ctrl+C to stop^)
"%VPY%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
if errorlevel 1 (
  echo [VOLSCAPE] ERROR: the app exited with an error. If port 8501 is busy, close the other app or run:
  echo            .venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
  exit /b 1
)
