import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
from keras.models import load_model

import xplique
# from xplique.plots import plot_attributions
from insight_face_models import *
from xplique.metrics import Deletion
from xplique.metrics import Insertion

from PIL import Image
import torchvision.transforms as transforms

from tqdm import tqdm

transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

def load_image(path, size=112):
    img = cv2.resize(cv2.imread(path)[...,::-1], (size, size))
    img = (img - 127.5) * 0.0078125
    return img.astype(np.float32)

def prepare_image(path, size=112):
    img = Image.open(path)
    img = transform(img).numpy()
    img = img.transpose(1, 2, 0)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='Deletion Metric')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/VGGFace2/test',
                        # default='datasets/celeb-a/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/VGGFace2/eval.txt',
                        # default='datasets/celeb-a/eval.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        default=2000,
                        help='Datasets.')
    parser.add_argument('--explanation-method', 
                        type=str, 
                        default='./explanation_results/vggface2/KernelShap',
                        # default='./explanation_results/celeba/SobolAttributionMethod',
                        help='Save path for saliency maps generated by interpretability methods.')
    parser.add_argument('--explanation-smdl', 
                        type=str, 
                        # default='./submodular_results/vggface2/grad-28x28-8/HsicAttributionMethod-97-1.0-1.0-1.0/npy',
                        default='./submodular_results/vggface2/grad-28x28-8/KernelShap-97-1.0-1.0-1.0/npy',
                        # default='./submodular_results/vggface2/grad-28x28-8/Rise-7x7-97-1.0-50.0-1.0/npy',
                        help='output directory to save results')
    parser.add_argument('--mode-data', 
                        type=str, 
                        default='VGGFace2',
                        # choices=['Celeb-A', "VGGFace2"],
                        help='')
    args = parser.parse_args()
    return args

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i
        
        single_mask = cv2.resize(single_mask, (7,7))    # for smooth
        single_mask = cv2.resize(single_mask, (112,112))
        
        batch_mask.append(single_mask)

    return np.array(batch_mask)

def main(args):
    if args.mode_data == "Celeb-A":
        keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-Celeb-A.h5"
        model = load_model(keras_model_path)
        print("load model {}".format(keras_model_path))
        class_number = 10177
        batch = 2048
    elif args.mode_data == "VGGFace2":
        keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-VGGFace2.h5"
        model = load_model(keras_model_path)
        print("load model {}".format(keras_model_path))
        class_number = 8631
        batch = 2048

    # data preproccess
    with open(args.eval_list, "r") as f:
        datas = f.read().split('\n')


    label = []
    input_image = []
    explanations = []
    smdl_mask = []
    for data in tqdm(datas[: args.eval_number]):
        label.append(int(data.strip().split(" ")[-1]))
        input_image.append(
            load_image(os.path.join(args.Datasets, data.split(" ")[0]))
        )
        explanations.append(
            np.load(
                os.path.join(args.explanation_method, data.split(" ")[0].replace(".jpg", ".npy")))
        )
        smdl_mask.append(
            np.load(
                os.path.join(args.explanation_smdl, data.split(" ")[0].replace(".jpg", ".npy")))    
        )
    label_onehot = tf.one_hot(np.array(label), class_number)
    input_image = np.array(input_image)
    explanations = np.array(explanations)
    smdl_mask = np.array(smdl_mask)

    # original
    # metric = Insertion(model, input_image, label_onehot, batch, baseline_mode=0.3, steps=7)
    metric = Insertion(model, input_image, label_onehot, batch)

    insertion_score_org = metric(explanations)
    
    batch_mask = convert_smdl_mask(smdl_mask)
    insertion_score = metric(batch_mask)
    print("Original Attribution Method Insertion Score: {}".format(insertion_score_org))
    print("Our Method Insertion Score: {}".format(insertion_score))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)