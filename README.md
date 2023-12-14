# Less is More: Fewer Interpretable Region via Submodular Subset Selection

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-green.svg?style=plastic)
![TensorFlow 2.12.0](https://img.shields.io/badge/tensorflow-2.12.0-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-Apache_2.0-green.svg?style=plastic)

![](./image/abstract.gif)

## Environment

```python
opencv-python
opencv-contrib-python
mtutils
```

```
conda create -n smdl python=3.10
conda activate smdl
python3 -m pip install tensorflow[and-cuda]

pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Some Sub-Region Division Method

|Sub-Region Division Method| Attribution Visualization | Org. Prediction Score | Highest Prediction Score | Insertion AUC Score | 
|:--:|:--:|:--:|:--:|:--:|
| SLICO | ![](image/slico.png) | 0.7262 | 0.9522 | 0.7604 |
| SEEDS | ![](image/seeds.png) | 0.7262 | 0.9918 | 0.8862 |
| Prior Saliency Map + Patch | ![](image/prior_saliency_division.png) | 0.7262 | 0.9710 | 0.7236 |
| Segment Anything Model | ![](image/sam.png) | 0.7262 | 0.9523 | 0.6803 |

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

## 👍 Acknowledgement

[Xplique](https://deel-ai.github.io/xplique/latest/): a Neural Networks Explainability Toolbox

[Score-CAM](https://github.com/tabayashi0117/Score-CAM/): a third-party implementation with Keras.