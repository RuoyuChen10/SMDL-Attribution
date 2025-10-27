@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."

:: 参数
set DATASETS=datasets\imagenet\ILSVRC2012_img_val
set EVAL_LIST=datasets\imagenet\val_clip_vitl_10.txt
set NUM=10
set L1=0.0
set L2=0.05
set L3=1.0
set L4=1.0

echo [1/4] Generate eval-list from actual files (NUM=%NUM%)...
python make_eval_list.py --datasets "%DATASETS%" --out "%EVAL_LIST%" --num %NUM%
if errorlevel 1 goto :fail

echo [2/4] Run submodular attribution (CLIP + superpixels)...
python -m submodular_attribution.smdl_explanation_imagenet_clip_superpixel ^
  --Datasets %DATASETS% ^
  --eval-list %EVAL_LIST% ^
  --lambda1 %L1% --lambda2 %L2% --lambda3 %L3% --lambda4 %L4% ^
  --begin 0 --end -1
if errorlevel 1 goto :fail

set EXP_DIR=submodular_results\imagenet-clip-vitl\slico-%L1%-%L2%-%L3%-%L4%
echo [3/4] Postprocess and evaluate AUC...
scripts\postprocess_results.cmd "%EXP_DIR%"
if errorlevel 1 goto :fail

echo [4/4] Done.
echo Results dir: %EXP_DIR%
echo Postprocess: %EXP_DIR%\postprocess
exit /b 0

:fail
echo ERROR occurred. Exit code=%errorlevel%
exit /b %errorlevel%
