# Minimal Interpretable Region Discovery via Submodular Subset Selection for Image Recognition

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-green.svg?style=plastic)
![TensorFlow 2.12.0](https://img.shields.io/badge/tensorflow-2.12.0-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-Apache_2.0-green.svg?style=plastic)

## 1. Generate saliency map

```
CUDA_VISIBLE_DEVICES=0 python generate_explanation_maps.py
```

## 2. Compute Minimal Interpretable Subset

```
CUDA_VISIBLE_DEVICES=0 python smdl_explanation.py
```
