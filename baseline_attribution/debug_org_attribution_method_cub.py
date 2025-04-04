import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from keras.models import load_model

# from keras.applications.efficientnet_v2 import preprocess_input
# from keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.vgg19 import preprocess_input

from tqdm import tqdm
import json
from utils import *

results_save_root = "./explanation_insertion_results"
explanation_method = "explanation_results/cub-resnet/KernelShap"
image_root_path = "datasets/CUB/test/"
if "resnet" in explanation_method:
    from keras.applications.resnet import (preprocess_input)
    keras_model_path = "ckpt/keras_model/cub-resnet101-new.h5"

eval_list = "datasets/CUB/eval_fair-resnet.txt"
save_doc = "cub-fair-resnet"
steps = 50

if "efficientnet" in save_doc:
    image_size_ = 384
else:
    image_size_ = 224

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

def main():
    mkdir(results_save_root)
    save_dir = os.path.join(results_save_root, save_doc)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, explanation_method.split("/")[-1])
    mkdir(save_dir)

    model = load_model(keras_model_path)

    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    for info in tqdm(infos):
        json_file = {}
        class_index = int(info.split(" ")[-1])
        image_path = os.path.join(image_root_path, info.split(" ")[0])

        save_class_path = os.path.join(save_dir, str(class_index))
        mkdir(save_class_path)

        mask_path = os.path.join(explanation_method, info.split(" ")[0].replace(".jpg", ".npy"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size_, image_size_))
        explanation = np.load(mask_path)

        insertion_explanation_images = []
        for i in range(1, steps+1):
            perturbed_rate = i / steps
            insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
        
        insertion_explanation_images_input = preprocess_input(
            np.array(insertion_explanation_images)[..., ::-1]
        )

        insertion_explanation_images_input_results = model(insertion_explanation_images_input)[:,class_index]
        json_file["recognition_score"] = insertion_explanation_images_input_results.numpy().tolist()

        with open(os.path.join(save_class_path, info.split(" ")[0].split("/")[-1].replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

    return

main()