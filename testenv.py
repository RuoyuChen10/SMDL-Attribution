# import torch; print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "num_gpus", torch.cuda.device_count())
# import cv2; assert hasattr(cv2, "ximgproc"), "opencv-contrib-python 未安装或版本不对"
# import numpy, scipy, skimage, sklearn, matplotlib, tqdm, imageio
# print("OK base libs")
# import open_clip, transformers; print("OK clip/trf")
# import segment_anything; print("OK SAM")
# import tensorflow as tf, tensorflow_probability as tfp, openturns as ot, xplique
# print("TF", tf.__version__, "Xplique OK")

# 在你当前运行的同一解释器里执行
modules = ["torch","torchvision","cv2","numpy","scipy","skimage","sklearn",
           "matplotlib","tqdm","imageio","open_clip","transformers",
           "segment_anything","tensorflow","tensorflow_probability","openturns","xplique"]
for m in modules:
    try:
        __import__(m)
        print("[OK]", m)
    except Exception as e:
        print("[MISS]", m, "->", e)
