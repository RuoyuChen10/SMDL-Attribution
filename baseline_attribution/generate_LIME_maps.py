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

from insight_face_models import *
from utils import *
from lime import lime_image

from lime.wrappers.scikit_image import SegmentationAlgorithm

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CUB"
net_mode  = "resnet" # "resnet", vgg

if mode == "CUB":
    if net_mode == "resnet":
        keras_model_path = "ckpt/keras_model/cub-resnet101.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval-resnet.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-resnet")
        from keras.applications.resnet import preprocess_input
    elif net_mode == "mobilenetv2":
        keras_model_path = "ckpt/keras_model/cub-mobilenetv2.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval-mobilenetv2.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-mobilenetv2")
        from keras.applications.mobilenet_v2 import preprocess_input
    dataset_path = "datasets/CUB/test"
    class_number = 200
    batch = 1
    
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
    batch = 1
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
    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=10,
                                                    max_dist=250, ratio=0.2)
    
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
    
    # explanation methods    
    explainer_method_name = "LIME"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for step in tqdm(range(total_steps), desc=explainer_method_name):
        image_name = input_data[step]
        
        X_raw = np.array(load_image(os.path.join(dataset_path, image_name)))
        y_label =  np.array(label[step])
        # Y_true = np.array(label[step * batch : step * batch + batch])
        # labels_ohe = tf.one_hot(Y_true, class_number)

        explanation = explainer.explain_instance(X_raw.astype('double'), model.predict, top_labels=5, hide_color=0, segmentation_fn=segmentation_fn, num_samples=100)
        # explanations = explainer(X_raw, labels_ohe)

        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 


        # if type(explanations) != np.ndarray:
        #     explanations = explanations.numpy()
        
        # for explanation, image_name, y_label in zip(explanations, image_names, Y_true):
        mkdir(os.path.join(exp_save_path, str(y_label)))
        np.save(os.path.join(exp_save_path, image_name.replace(".jpg", "")), heatmap)

    return

main()