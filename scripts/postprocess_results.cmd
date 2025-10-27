@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."

if "%~1"=="" (
  echo Usage: scripts\postprocess_results.cmd ^<explanation-dir^>
  echo Example:
  echo   scripts\postprocess_results.cmd submodular_results\imagenet-clip-vitl\slico-0.0-0.05-1.0-1.0
  exit /b 1
)

set EXP_DIR=%~1

echo [1/2] Postprocessing results under "%EXP_DIR%" ...
python tools\postprocess_submodular_results.py --explanation-dir "%EXP_DIR%" --gif --gif-steps 0
if errorlevel 1 goto :fail

echo [2/2] Done. See "%EXP_DIR%\postprocess" for outputs.
exit /b 0

:fail
echo ERROR occurred. Exit code=%errorlevel%
exit /b %errorlevel% 