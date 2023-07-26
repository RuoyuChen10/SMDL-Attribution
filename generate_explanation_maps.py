# -*- coding: utf-8 -*-  

"""
Created on 2023/6/28

@author: Ruoyu Chen
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

mode = "VGGFace2"

if mode == "Celeb-A":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-Celeb-A.h5"
    dataset_path = "datasets/celeb-a/test"
    dataset_index = "datasets/celeb-a/test.txt"
    class_number = 10177
    batch = 256
    SAVE_PATH = os.path.join(SAVE_PATH, "celeba")
    mkdir(SAVE_PATH)

elif mode == "VGGFace2":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-VGGFace2.h5"
    dataset_path = "datasets/VGGFace2/test"
    dataset_index = "datasets/VGGFace2/test.txt"
    class_number = 8631
    batch = 2048
    SAVE_PATH = os.path.join(SAVE_PATH, "vggface2")
    mkdir(SAVE_PATH)

def load_image(path, size=112):
    img = cv2.resize(cv2.imread(path)[...,::-1], (size, size))
    img = (img - 127.5) * 0.0078125
    return img.astype(np.float32)

def main():
    # Load model
    model = load_model(keras_model_path)
    
    model.layers[-1].activation = tf.keras.activations.linear
    batch_size = 4096
    
    # define explainers
    explainers = [
        # Saliency(model),
        # GradientInput(model),
        # GuidedBackprop(model),
        # IntegratedGradients(model, steps=80, batch_size=batch_size),
        # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(model, nb_samples=80, batch_size=batch_size),
        # VarGrad(model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # Rise(model, nb_samples=500, batch_size=batch_size),
        # SobolAttributionMethod(model, batch_size=batch_size),
        HsicAttributionMethod(model, batch_size=batch_size),
        Lime(model, nb_samples = 1000),
        KernelShap(model, nb_samples = 1000)
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