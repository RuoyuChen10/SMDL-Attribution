# -*- coding: utf-8 -*-  

"""
Created on 2024/6/6

@author: Ruoyu Chen
Reference: https://github.com/vaynexie/CausalX-ViT
"""

import os

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from open_clip import create_model_from_pretrained, get_tokenizer

from .ViT_CX.ViT_CX import ViT_CX

import torch
from torchvision import transforms

from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "Quilt"
net_mode  = "Quilt" # "resnet", vgg

if mode == "Quilt":
    if net_mode == "Quilt":
        img_size = 224
        dataset_index = "datasets/medical_lung/LC25000_lung_quilt_1k_false.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "lung-quilt-false")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/medical_lung/lung_dataset"
    class_number = 3
    batch = 100
    mkdir(SAVE_PATH)

class QuiltModel_Super(torch.nn.Module):
    def __init__(self, 
                 download_root=".checkpoints/QuiltNet-B-32",
                 device = "cuda"):
        super().__init__()
        self.model, _ = create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32', cache_dir=download_root)
        self.device = device
            
    def forward(self, vision_inputs):
        
        # with torch.no_grad():
        image_features = self.model.encode_image(vision_inputs)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.semantic_feature.T).softmax(dim=-1)
        return scores.float()

data_transform = transforms.Compose(
        [
            transforms.Resize(
                (224,224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def preprocess_image(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image = data_transform(image)
    return image.unsqueeze(0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    vis_model = QuiltModel_Super()
    vis_model.eval()
    vis_model.to(device)
    print("load Quilt-1M model")
    
    tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    texts = tokenizer([lc_lung_template + l for l in lc_lung_classes], context_length=77).to(device)

    with torch.no_grad():
        semantic_feature = vis_model.model.encode_text(texts) * 10
    vis_model.semantic_feature = semantic_feature
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        label.append(int(data.strip().split(" ")[-1]))
        input_data.append(
            os.path.join(dataset_path, data.split(" ")[0])
        )
    
    # explanation methods    
    explainer_method_name = "ViT-CX"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    target_layer=vis_model.model.visual.transformer.resblocks[-1].ln_1
    
    for step in tqdm(range(len(input_data))):
        img_path = input_data[step]
        category = label[step]
        
        image = cv2.imread(img_path)
        image_cpu = preprocess_image(image)
        
        # Perform ViT-CX
        result=ViT_CX(vis_model, 
                      image_cpu, 
                      target_layer, 
                      target_category=category, 
                      distance_threshold=0.1, 
                      gpu_batch=50)
        
        np.save(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".jpeg", "")), result)

    return

main()