# requires -version 5.1
$ErrorActionPreference = "Stop"

# 工作目录设为脚本所在目录的上级（项目根）
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Host "[1/4] 准备最小数据集与评测列表..."
$imgDir = Join-Path $ROOT "datasets/imagenet/ILSVRC2012_img_val"
New-Item -ItemType Directory -Path $imgDir -Force | Out-Null
Copy-Item -Path (Join-Path $ROOT "examples/dog_image.jpg") -Destination (Join-Path $imgDir "ILSVRC2012_val_00000001.JPEG") -Force

$evalListDir = Join-Path $ROOT "datasets/imagenet"
New-Item -ItemType Directory -Path $evalListDir -Force | Out-Null
$evalList = Join-Path $evalListDir "val_clip_vitl_1.txt"
"ILSVRC2012_val_00000001.JPEG 207" | Out-File -FilePath $evalList -Encoding ascii -Force

Write-Host "[2/4] 运行子模归因（CLIP + 超像素，单卡）..."
$env:PYTHONUNBUFFERED = "1"
python -m submodular_attribution.smdl_explanation_imagenet_clip_superpixel --Datasets "$imgDir" --eval-list "$evalList" --lambda1 0 --lambda2 0.05 --lambda3 1 --lambda4 1 --begin 0 --end -1
if ($LASTEXITCODE -ne 0) { throw "子模归因执行失败，退出码 $LASTEXITCODE" }

Write-Host "[3/4] 运行评估（Insertion/Deletion AUC）..."
$expDir = Join-Path $ROOT "submodular_results/imagenet-clip-vitl/slico-0.0-0.05-1.0-1.0"
python -m evals.eval_AUC_faithfulness --explanation-dir "$expDir"
if ($LASTEXITCODE -ne 0) { throw "评估脚本执行失败，退出码 $LASTEXITCODE" }

Write-Host "[4/4] 完成。结果目录：" $expDir 