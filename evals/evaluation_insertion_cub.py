import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
from keras.models import load_model
from keras.applications.resnet import (preprocess_input)
# from keras.applications.efficientnet_v2 import preprocess_input
# from keras.applications.vgg19 import preprocess_input
# from keras.applications.mobilenet_v2 import preprocess_input

import xplique
# from xplique.plots import plot_attributions
from insight_face_models import *
from xplique.metrics import Insertion

from PIL import Image

from tqdm import tqdm

img_size = 224
# img_size = 384

def load_image(path, size=384):
    img = cv2.resize(cv2.imread(path)[...,::-1], (img_size, img_size))
    img = preprocess_input(np.array(img))
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='Insertion Metric')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/CUB/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/CUB/eval-resnet.txt',
                        # default='datasets/CUB/eval_fair-vgg19.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        # default=1,
                        default=600,
                        help='Datasets.')
    parser.add_argument('--explanation-method', 
                        type=str, 
                        default='./explanation_results/cub-resnet/KernelShap',
                        help='Save path for saliency maps generated by interpretability methods.')
    parser.add_argument('--explanation-smdl', 
                        type=str, 
                        # default='./submodular_results/cub-resnet/superpixel-seeds-1.0-1.0-1.0-1.0/npy',
                        default='./submodular_results-v1-iclr-results/cub/grad-10x10-4/HsicAttributionMethod-24-1.0-1.0-1.0-1.0/npy',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i

        # single_mask = cv2.resize(single_mask.astype(np.uint8), (10,10))    # for smooth
        # single_mask = cv2.resize(single_mask.astype(np.uint8), (img_size,img_size))
        # single_mask = np.exp(single_mask / single_mask.max() / 0.1)
        # single_mask = (single_mask / single_mask.max())*255
        
        batch_mask.append(single_mask)

    return np.array(batch_mask).mean(-1)

def main(args):
    if "resnet" in args.explanation_method:
        keras_model_path = "ckpt/keras_model/cub-resnet101.h5"
    elif "vgg19" in args.explanation_method:
        keras_model_path = "ckpt/keras_model/cub-vgg19.h5"
    elif "efficientnet" in args.explanation_method:
        keras_model_path = "ckpt/keras_model/cub-efficientnetv2m.h5"
    elif "mobilenetv2" in args.explanation_method:
        keras_model_path = "ckpt/keras_model/cub-mobilenetv2.h5"
    else:
        keras_model_path = "ckpt/keras_model/cub-resnet101.h5"
    print(keras_model_path)
    model = load_model(keras_model_path)
    # model.layers[-1].activation = tf.keras.activations.linear
    class_number = 200
    batch = 256

    # data preproccess
    with open(args.eval_list, "r") as f:
        datas = f.read().split('\n')

    label = []
    input_image = []
    explanations = []
    # smdl_mask = []
    for data in tqdm(datas[: args.eval_number]):
        label.append(int(data.strip().split(" ")[-1]))
        input_image.append(
            load_image(os.path.join(args.Datasets, data.split(" ")[0]))
        )
        explanations.append(
            np.load(
                os.path.join(args.explanation_method, data.split(" ")[0].replace(".jpg", ".npy")))
        )
        # smdl_mask.append(
        #     np.load(
        #         os.path.join(args.explanation_smdl, data.split(" ")[0].replace(".jpg", ".npy")))    
        # )
    label_onehot = tf.one_hot(np.array(label), class_number)
    input_image = np.array(input_image)
    explanations = np.array(explanations)
    # smdl_mask = np.array(smdl_mask)

    # original
    # metric = Insertion(model, input_image, label_onehot, batch, baseline_mode=0.3, steps=7)
    metric = Insertion(model, input_image, label_onehot, batch, steps=25)   #, baseline_mode=np.array([[[23.060997 , 10.221001 ,  3.3199997]]])

    insertion_score_org = metric(explanations)
    
    # batch_mask = convert_smdl_mask(smdl_mask)
    # insertion_score = metric(batch_mask)
    print("Original Attribution Method Insertion Score: {}".format(insertion_score_org))
    # print("Our Method Insertion Score: {}".format(insertion_score))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)