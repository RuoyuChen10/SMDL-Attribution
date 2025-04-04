# -*- coding: utf-8 -*-  

"""
Created on 2023/6/28

@author: Ruoyu Chen
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import math
import tensorflow_addons as tfa
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm

from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

from insight_face_models import *
from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CUB"
net_mode  = "resnet" # "resnet", vgg

if mode == "Celeb-A":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-Celeb-A.h5"
    dataset_path = "datasets/celeb-a/test"
    dataset_index = "datasets/celeb-a/test.txt"
    class_number = 10177
    batch = 256
    img_size = 112
    SAVE_PATH = os.path.join(SAVE_PATH, "celeba")
    mkdir(SAVE_PATH)

elif mode == "VGGFace2":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-VGGFace2.h5"
    dataset_path = "datasets/VGGFace2/test"
    dataset_index = "datasets/VGGFace2/eval.txt"
    class_number = 8631
    batch = 2048
    img_size = 112
    SAVE_PATH = os.path.join(SAVE_PATH, "vggface2")
    mkdir(SAVE_PATH)

elif mode == "CUB":
    if net_mode == "resnet":
        keras_model_path = "ckpt/keras_model/cub-resnet101-new.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval-resnet.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-resnet")
        from keras.applications.resnet import preprocess_input
    elif net_mode == "mobilenetv2":
        keras_model_path = "ckpt/keras_model/cub-mobilenetv2.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval-resnet.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-resnet")
        from keras.applications.mobilenet_v2 import preprocess_input
    dataset_path = "datasets/CUB/test"
    class_number = 200
    batch = 100
    
    mkdir(SAVE_PATH)

elif mode == "CUB-FAIR":
    if net_mode == "resnet":
        keras_model_path = "ckpt/keras_model/cub-resnet101-new.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-resnet.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-resnet")
        from keras.applications.resnet import preprocess_input
    elif net_mode == "efficientnet":
        keras_model_path = "ckpt/keras_model/cub-efficientnetv2m.h5"
        img_size = 384
        dataset_index = "datasets/CUB/eval_fair-efficientnet.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-efficientnet")
        from keras.applications.efficientnet_v2 import preprocess_input
    elif net_mode == "vgg":
        keras_model_path = "ckpt/keras_model/cub-vgg19.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-vgg19.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-vgg19")
        from keras.applications.vgg19 import preprocess_input
    elif net_mode == "mobilenetv2":
        keras_model_path = "ckpt/keras_model/cub-mobilenetv2.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-mobilenetv2.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-mobilenetv2")
        from keras.applications.mobilenet_v2 import preprocess_input

    dataset_path = "datasets/CUB/test"
    class_number = 200
    batch = 100
    mkdir(SAVE_PATH)

elif mode == "CUB-CROP":
    keras_model_path = "ckpt/keras_model/cub-resnet101-crop.h5"
    dataset_path = "datasets/CUB/test_crop"
    dataset_index = "datasets/CUB/eval_crop_random.txt"
    class_number = 200
    batch = 100
    img_size = 224
    SAVE_PATH = os.path.join(SAVE_PATH, "cub_crop")
    mkdir(SAVE_PATH)

def load_image(path):
    img = cv2.resize(cv2.imread(path)[...,::-1], (img_size, img_size))
    if mode == "VGGFace2" or mode == "Celeb-A":
        img = (img - 127.5) * 0.0078125
        return img.astype(np.float32)
    elif mode == "CUB" or mode == "CUB-CROP" or mode == "CUB-FAIR":
        img = preprocess_input(np.array(img))
        return img

def main():
    # Load model
    model = load_model(keras_model_path)
    # model.layers[-1].activation = tf.keras.activations.linear
    batch_size = 256
    
    # define explainers
    explainers = [
        # Saliency(model),
        # GradientInput(model),
        # GuidedBackprop(model),
        IntegratedGradients(model, steps=80, batch_size=64),
        # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(model, nb_samples=80, batch_size=batch_size),
        # VarGrad(model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # GradCAMPP(model),
        # Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # Rise(model, nb_samples=4000, batch_size=batch_size),
        # SobolAttributionMethod(model, batch_size=batch_size),
        # HsicAttributionMethod(model, batch_size=batch_size),
        # Lime(model, nb_samples = 1000),
        # KernelShap(model, nb_samples = 1000, batch_size=32)
    ]
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        label.append(int(data.strip().split(" ")[-1]))
        input_data.append(
            data.split(" ")[0]
        )
    
    total_steps = math.ceil(len(input_data) / batch)
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            image_names = input_data[step * batch : step * batch + batch]
            
            if os.path.exists(
                os.path.join(exp_save_path, image_names[0].replace(".jpg", ".npy"))
            ):
                print(1)
                continue
            X_raw = np.array([load_image(os.path.join(dataset_path, image_name)) for image_name in image_names])

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = tf.one_hot(Y_true, class_number)
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name, y_label in zip(explanations, image_names, Y_true):
                mkdir(os.path.join(exp_save_path, str(y_label)))
                np.save(os.path.join(exp_save_path, image_name.replace(".jpg", "")), explanation)
    
    return

main()
