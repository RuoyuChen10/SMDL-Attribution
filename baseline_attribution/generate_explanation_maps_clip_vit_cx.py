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

import clip

from .ViT_CX.ViT_CX import ViT_CX

import torch
from torchvision import transforms

from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CLIP"
net_mode  = "CLIP" # "resnet", vgg

if mode == "CLIP":
    if net_mode == "CLIP":
        img_size = 224
        dataset_index = "datasets/imagenet/val_clip_vitl_2k_false.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-false")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 100
    mkdir(SAVE_PATH)

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def equip_semantic_modal(self, modal_list):
        text = clip.tokenize(modal_list).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
    def forward(self, vision_inputs):
        
        # with torch.no_grad():
        image_features = self.model.encode_image(vision_inputs)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.text_features.T).softmax(dim=-1)
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
    vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP")
    vis_model.eval()
    vis_model.to(device)
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    vis_model.text_features = semantic_feature
    
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
        
        np.save(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".JPEG", "")), result)

    return

main()