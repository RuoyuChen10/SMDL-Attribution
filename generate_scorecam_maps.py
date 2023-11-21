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
from tensorflow.keras.models import Model
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm

from insight_face_models import *
from utils import *

# from keras.applications.resnet import (
#     ResNet50, ResNet101, preprocess_input, decode_predictions)
# from keras.applications.efficientnet_v2 import preprocess_input

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CUB-FAIR"
net_mode  = "vgg" # "resnet", vgg

if mode == "Celeb-A":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-Celeb-A.h5"
    dataset_path = "datasets/celeb-a/test"
    dataset_index = "datasets/celeb-a/test.txt"
    class_number = 10177
    batch = 1
    img_size = 112
    SAVE_PATH = os.path.join(SAVE_PATH, "celeba")
    mkdir(SAVE_PATH)

elif mode == "VGGFace2":
    keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-VGGFace2.h5"
    dataset_path = "datasets/VGGFace2/test"
    dataset_index = "datasets/VGGFace2/eval.txt"
    class_number = 8631
    batch = 1
    img_size = 112
    SAVE_PATH = os.path.join(SAVE_PATH, "vggface2")
    mkdir(SAVE_PATH)

elif mode == "CUB":
    keras_model_path = "ckpt/keras_model/cub-resnet101-new.h5"
    dataset_path = "datasets/CUB/test"
    dataset_index = "datasets/CUB/eval.txt"
    class_number = 200
    img_size = 224
    batch = 1
    layer_name = "conv5_block3_3_conv"
    SAVE_PATH = os.path.join(SAVE_PATH, "cub")
    mkdir(SAVE_PATH)

elif mode == "CUB-FAIR":
    if net_mode == "resnet":
        keras_model_path = "ckpt/keras_model/cub-resnet101-new.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-resnet.txt"
        layer_name = "conv5_block3_3_conv"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub")
        from keras.applications.resnet import preprocess_input
    elif net_mode == "efficientnet":
        keras_model_path = "ckpt/keras_model/cub-efficientnetv2m.h5"
        img_size = 384
        dataset_index = "datasets/CUB/eval_fair-efficientnet.txt"
        layer_name = "top_conv"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-efficientnet")
        from keras.applications.efficientnet_v2 import preprocess_input
    elif net_mode == "vgg":
        keras_model_path = "ckpt/keras_model/cub-vgg19.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-vgg19.txt"
        layer_name = "block5_conv4"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-vgg19")
        from keras.applications.vgg19 import preprocess_input
    elif net_mode == "mobilenetv2":
        keras_model_path = "ckpt/keras_model/cub-mobilenetv2.h5"
        img_size = 224
        dataset_index = "datasets/CUB/eval_fair-mobilenetv2.txt"
        layer_name = "Conv_1"
        SAVE_PATH = os.path.join(SAVE_PATH, "cub-mobilenetv2")
        from keras.applications.mobilenet_v2 import preprocess_input
    dataset_path = "datasets/CUB/test"
    class_number = 200
    batch = 100
    mkdir(SAVE_PATH)

def load_image(path):
    img = cv2.resize(cv2.imread(path)[...,::-1], (img_size, img_size))
    if mode == "VGGFace2" or mode == "Celeb-A":
        img = (img - 127.5) * 0.0078125
        return img.astype(np.float32)
    elif mode == "CUB" or mode == "CUB-CROP" or mode == "CUB-FAIR":
        img = preprocess_input(np.array(img))
        return img
    
class ScoreCAM(object):
    """
    refer from repo: https://github.com/tabayashi0117/Score-CAM/
    """
    def __init__(self,
                 model,
                 layer_name= None):
        self.model = model
        self.layer_name = layer_name

    def inference(self, image, max_N=-1):
        cls = np.argmax(self.model.predict(image))
        act_map_array = Model(inputs=self.model.input, outputs=self.model.get_layer(self.layer_name).output).predict(image)

        # extract effective maps
        if max_N != -1:
            act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            act_map_array = act_map_array[:,:,:,max_N_indices]
        
        input_shape = self.model.layers[0].output_shape[0][1:]  # get input shape
        # 1. upsample to original input size
        act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
        # 2. normalize the raw activation value in each activation map into [0, 1]
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)
        # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(image).astype(np.float32)
            for k in range(3):
                masked_input[0,:,:,k] *= act_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)
        # 4. feed masked inputs into CNN model and softmax
        pred_from_masked_input_array = self.model.predict(masked_input_array)
        # 5. define weight as the score of target class
        weights = pred_from_masked_input_array[:,cls]
        # 6. get final class discriminative localization map as linear weighted combination of all activation maps
        cam = np.dot(act_map_array[0,:,:,:], weights)
        cam = np.maximum(0, cam)  # Passing through ReLU
        cam /= np.max(cam)  # scale 0 to 1.0

        return cam

def main():
    # Load model
    model = load_model(keras_model_path)
    print(model.summary())
    
    # define explainers
    explainer = ScoreCAM(model, layer_name)
    
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
    
    # explanation methods    
    explainer_method_name = "ScoreCAM"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    for image_name, y_label in tqdm(zip(input_data, label)):

        if os.path.exists(
            os.path.join(exp_save_path, image_name.replace(".jpg", ".npy"))
        ):
            print(1)
            continue
        
        X_raw = np.array([load_image(os.path.join(dataset_path, image_name))])
        # Y_true = np.array(label[step * batch : step * batch + batch])
        # labels_ohe = tf.one_hot(Y_true, class_number)
        
        explanation = explainer.inference(X_raw)
        explanation = cv2.resize(explanation, (img_size, img_size))
        
        mkdir(os.path.join(exp_save_path, str(y_label)))
        np.save(os.path.join(exp_save_path, image_name.replace(".jpg", "")), explanation)
    
    return

main()