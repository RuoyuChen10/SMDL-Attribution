@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0\.."

echo [1/4] Preparing minimal dataset and eval list...
mkdir datasets\imagenet\ILSVRC2012_img_val 2>nul
copy /Y examples\dog_image.jpg datasets\imagenet\ILSVRC2012_img_val\ILSVRC2012_val_00000001.JPEG >nul
mkdir datasets\imagenet 2>nul
echo ILSVRC2012_val_00000001.JPEG 207> datasets\imagenet\val_clip_vitl_1.txt

echo [2/4] Running submodular attribution (CLIP + superpixels, single GPU)...
set PYTHONUNBUFFERED=1
python -m submodular_attribution.smdl_explanation_imagenet_clip_superpixel ^
  --Datasets datasets/imagenet/ILSVRC2012_img_val ^
  --eval-list datasets/imagenet/val_clip_vitl_1.txt ^
  --lambda1 0 --lambda2 0.05 --lambda3 1 --lambda4 1 ^
  --begin 0 --end -1
if errorlevel 1 goto :fail

echo [3/4] Evaluating Insertion/Deletion AUC...
set EXP_DIR=submodular_results\imagenet-clip-vitl\slico-0.0-0.05-1.0-1.0
python -m evals.eval_AUC_faithfulness --explanation-dir "%EXP_DIR%"
if errorlevel 1 goto :fail

echo [4/4] Done. Results in: %EXP_DIR%
goto :eof

:fail
echo ERROR occurred. Exit code=%errorlevel%
exit /b %errorlevel%