# SMDL-Attribution 项目总览

> ICLR 2024 Oral: Less is More: Fewer Interpretable Region via Submodular Subset Selection

本仓库实现了基于子模集合选择（Submodular Subset Selection, SMDL）的高效可解释性归因方法，支持多模态/多框架（主要聚焦 PyTorch，部分历史代码依赖 TensorFlow/Keras），并提供了评测与可视化工具、脚本及教程。

## 1. 顶层运行逻辑
- **输入**: 待解释的样本（图像/音频等）与目标模型（分类、检索或多模态基础模型如 CLIP、ImageBind、LanguageBind、Quilt 等）。
- **子区域划分**: 使用超像素（SLICO/SEEDS）或 SAM 生成候选子区域集合 `V`。
- **黑盒打分**: 在遮挡/还原等扰动下，调用黑盒模型前向得到置信度变化，构造一致性/协作性等子模评分曲线。
- **子模优化**: 基于子模目标（例如边际增益最大化）进行贪心选择，得到少量最解释性的子区域子集 `S ⊂ V`。
- **保存中间结果**: 将子集序列、对应置信度曲线、可视化图、以及每步增量区域 `npy` 保存至 `submodular_results/...`。
- **评估**: 使用 Insertion/Deletion AUC、MuFidelity 等指标对归因结果进行定量评估。

## 2. 目录与文件功能
- `README.md`: 项目简介与最新动态、环境与模型下载、基本运行示例。
- `requirement.txt`: 依赖清单（Python 包）参考。
- `utils.py`:
  - `SubRegionDivision(image, mode, region_size)`: 超像素划分（SLICO/SEEDS），返回子区域像素集合。
  - 图像读写、标准化与可视化（`load_image`, `gen_cam`, `norm_image` 等）。
  - 常见类别与模板（ImageNet/CUB/VGGSOUND 等）文本提示集合（用于 CLIP/LanguageBind 等检索式模型）。
- `xplique_addons.py`:
  - 对 Xplique 的扩展（基于 Sobol/HSIC 的灵敏度归因范式），包含蒙特卡洛采样器（QMC、LHS 等）与核函数统计量的实现；主要面向 Keras/TensorFlow 流程的全局敏感度分析。
- `scripts/`:
  - 多卡并行脚本，如 `clip_multigpu.sh`：将评测列表分片分配至多张 GPU，调用 `submodular_attribution.smdl_explanation_imagenet_clip_superpixel` 生成归因结果。
- `submodular_attribution/`:
  - 各任务/骨干的子模归因主程序（如 `smdl_explanation_*` Python 模块与若干模型封装），实现从数据读取、子区域生成、打分、贪心选择、结果保存的端到端流程。
  - 注：部分具体文件名在清单中可见（如 `submodular_vit_efficient.py`, `submodular_single_modal.py`, `submodular_cub*.py` 等），用于不同数据集与模型的子模流程封装。
- `baseline_attribution/`:
  - 基线方法与先验显著图生成脚本（Grad/Score-CAM/LIME、以及 ViT/CLIP/ImageBind/LanguageBind/Quilt 的显著性图管线）。
  - 例如 `generate_explanation_maps_imagebind.py`, `generate_scorecam_maps.py` 等。
- `evals/`:
  - 评估脚本：Insertion/Deletion、MuFidelity、错误分析等。
  - 示例：`eval_AUC_faithfulness.py` 从 `submodular_results/.../{json,npy}` 读取一致性/协作性曲线与区域序列，计算 AUC 指标。
- `visualization/`:
  - Demo 脚本：`demo.py`, `demo-smooth.py` 等，用于渲染归因热力图/叠加图。
- `SAM_mask_generate.py`:
  - 基于 Segment Anything（SAM）的掩码自动生成，`processing_sam_concepts` 负责掩码去交叠与筛选，输出 `numpy` 区域集合。
- `insight_face_models.py`:
  - 人脸识别/表征模型的 Keras 封装与构建（多主干，如 ResNet/EfficientNet/MobileNet 等），提供嵌入头（E/GAP/GDC）。
- `datasets/`、`configs/`、`models/`、`ckpt/`、`mtcnn/`:
  - 数据与配置组织、模型文件与第三方模块（人脸预处理等）。
- `tools/`:
  - 对齐、赋值等工具函数：`alignment.py`, `assigned_value.py`。
- `tutorial/`:
  - 各模型/模态的 Notebook 教程（CUB、CLIP、ImageBind、LanguageBind、Quilt、VGGSound 等）。

## 3. 子模归因主流程（高层）
以超像素为例（脚本形如 `smdl_explanation_*_superpixel.py`）：
1. 解析参数（数据根目录、评测列表、超像素模式、子模超参数 `lambda1..lambda4`、处理区间等）。
2. 读取评测列表（如 ImageNet 的 `val_clip_vitl_5k_true.txt`），逐条载入样本与标签/文本提示。
3. 子区域生成：
   - `utils.SubRegionDivision(image, mode="slico"|"seeds", region_size=...)` 或 `SAM_mask_generate.py` 预生成的 `npy` 区域集合。
4. 黑盒打分：
   - 对候选区域进行遮挡/还原组合，调用任务模型（分类器/检索模型）前向，记录目标类别分数或相似度变化，形成一致性/协作性曲线。
5. 子模贪心：
   - 以边际增益为准则，迭代选择若干子区域，得到解释序列与每步累计区域图。
6. 保存结果：
   - `json/`：曲线分数、元数据；`npy/`：每步增量区域；`vis/`：覆盖图/热图。

## 4. 评估指标与脚本
- Insertion/Deletion AUC（`evals/eval_AUC_faithfulness.py`）
  - 将区域按解释序列逐步插入（Insertion）或删除（Deletion），计算曲线下面积（AUC）。
- MuFidelity（`evals/evaluation_mufidelity_*.py`）
  - 衡量解释与模型行为的一致性鲁棒性。
- 错误调试（`evaluation_mistake_debug_*.py`）
  - 基于解释结果识别模型错误或偏差来源。

## 5. 基线与先验显著图
- `baseline_attribution/generate_*.py` 系列：
  - Grad、Score-CAM、LIME、以及针对 CLIP/ImageBind/LanguageBind/Quilt 的显著图生成。
  - 输出通常作为“先验”或对比方法，与 SMDL 组合为“先验显著图 + Patch”子区域划分方案。

## 6. 多卡并行与批处理
- `scripts/clip_multigpu.sh`：
  - 按行切分评测列表，分配至多 GPU 并行运行：
  - 关键参数：`--lambda1..4`（子模权重）、`--begin/--end`（行区间）。

## 7. 可视化
- `visualization/demo.py`, `demo-smooth.py`: 读取保存的 `npy/json`，生成叠加可视化图；
- 也可使用 `utils.gen_cam` 叠加热力图到原图上。

## 8. 环境与依赖
- 主要依赖：OpenCV、NumPy、SciPy、scikit-image/learn、Matplotlib、tqdm、xplique、segment-anything 等。
- PyTorch 为主，历史 Keras/TensorFlow 代码用于部分模型与 HSIC/Sobol 归因实验。
- 参考 `README.md` 与 `requirement.txt` 安装，SAM 需额外安装其官方依赖。

## 9. 数据与模型
- `ckpt/keras_model` 与 `ckpt/pytorch_model`：README 提供 HuggingFace 下载链接。
- 数据组织示例：`datasets/imagenet/ILSVRC2012_img_val`、CUB、VGGFace2、Celeb-A、VGGSound 等。

## 10. 一键示例（ImageNet + CLIP，多卡）
```bash
./scripts/clip_multigpu.sh
# 运行后结果位于：submodular_results/imagenet-clip-vitl/slico-0.0-0.05-1.0-1.0
python -m evals.eval_AUC_faithfulness --explanation-dir submodular_results/imagenet-clip-vitl/slico-0.0-0.05-1.0-1.0
```

## 11. 注意事项
- 超像素依赖 `opencv-contrib-python`（ximgproc）；SAM 推理需较大显存。
- 归因与评估读取/保存路径需与脚本参数一致；多卡脚本的 `--begin/--end` 为行号区间。
- 统一使用 `numpy` 保存区域序列，评估脚本按 `json/npy` 配对读取。

## 12. 扩展与二次开发建议
- 新模型接入：
  - 封装“前向打分”接口，输出目标分数或相似度；
  - 适配子区域生成（超像素/SAM/先验显著图 + Patch）。
- 新指标接入：
  - 参考 `evals/` 模板，读取 `json/npy`，实现曲线构造与 AUC/统计量计算。
- 性能优化：
  - 使用多卡/批处理；缓存模型前向；减少冗余区域评估；对大图像启用区域过滤阈值。 