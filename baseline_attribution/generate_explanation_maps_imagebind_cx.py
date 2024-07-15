# -*- coding: utf-8 -*-  

"""
Created on 2024/6/20

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

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from .ViT_CX.ViT_CX import ViT_CX

import torch
from torchvision import transforms

from utils import *

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "imagenet"
net_mode  = "imagebind" # "resnet", vgg

if mode == "imagenet":
    if net_mode == "imagebind":
        img_size = 224
        dataset_index = "datasets/imagenet/val_imagebind_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-imagebind-true")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 100
    mkdir(SAVE_PATH)
    
class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
    def mode_selection(self, mode):
        if mode not in ["text", "audio", "thermal", "depth", "imu"]:
            print("mode {} does not comply with the specification, please select from \"text\", \"audio\", \"thermal\", \"depth\", \"imu\".".format(mode))
        else:
            self.mode = mode
            print("Select mode {}".format(mode))
            
    def equip_semantic_modal(self, modal_list):
        if self.mode == "text":
            self.semantic_modal = data.load_and_transform_text(modal_list, self.device)
        elif self.mode == "audio":
            self.semantic_modal = data.load_and_transform_audio_data(modal_list, self.device)
        
        input = {
                # "vision": vision_inputs,
                self.mode: self.semantic_modal
            }
        with torch.no_grad():
            self.semantic_modal = self.base_model(input)[self.mode]
        print("Equip with {} modal.".format(self.mode))
        
    def forward(self, vision_inputs):
        inputs = {
            "vision": vision_inputs,
        }
        
        # with torch.no_grad():
        embeddings = self.base_model(inputs)
        
        scores = torch.softmax(embeddings["vision"] @ self.semantic_modal.T, dim=-1)
        return scores

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
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    vis_model = ImageBindModel_Super(model, device)
    vis_model.mode_selection("text")
    
    semantic_path = "ckpt/semantic_features/imagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    vis_model.semantic_modal = semantic_feature
    
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
    
    target_layer=vis_model.base_model.modality_trunks.vision.blocks[-1].norm_1
    
    
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