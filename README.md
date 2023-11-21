# Less is More: Fewer Interpretable Region via Submodular Subset Selection

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-green.svg?style=plastic)
![TensorFlow 2.12.0](https://img.shields.io/badge/tensorflow-2.12.0-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-Apache_2.0-green.svg?style=plastic)

![](./image/abstract.gif)

## 1. Generate saliency map

First, the priori saliency maps for sub-region division needs to be generated.

```
CUDA_VISIBLE_DEVICES=0 python generate_explanation_maps.py
```

Don't forget to open this file and revise the variable `mode` and `net_mode`:

- `mode`: ["Celeb-A", "VGGFace2", "CUB", "CUB-FAIR"]

- `net_mode`: ["resnet", "efficientnet", "vgg19", "mobilenetv2"], note that these net_mode only for `mode` is CUB-FAIR.



## 2. Compute Minimal Interpretable Subset

```
CUDA_VISIBLE_DEVICES=0 python smdl_explanation.py
```
