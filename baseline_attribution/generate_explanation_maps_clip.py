# -*- coding: utf-8 -*-  

"""
Created on 2024/6/3

@author: Ruoyu Chen
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import clip

from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import torch
from torchvision import transforms

import tensorflow as tf
from utils import *

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "CLIP"
net_mode  = "CLIP" # "resnet", vgg

if mode == "CLIP":
    if net_mode == "CLIP":
        img_size = 224
        dataset_index = "datasets/imagenet/val_clip_vitl_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-clip-vitl-true")
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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
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

def load_and_transform_vision_data(image_paths, device, channel_first=False):
    if image_paths is None:
        return None

    image_outputs = []
    
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    image_outputs = torch.stack(image_outputs, dim=0)
    if channel_first:
        pass
    else:
        image_outputs = image_outputs.permute(0,2,3,1)
    return image_outputs.cpu().numpy()   

def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            
            with torch.no_grad():
                class_embeddings = model.model.encode_text(texts)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).cuda()
    return zeroshot_weights*100

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
    
    wrapped_model = TorchWrapper(vis_model.eval(), device)
    
    batch_size = 32
    
    # define explainers
    explainers = [
        # Saliency(model),
        # GradientInput(model),
        # GuidedBackprop(model),
        IntegratedGradients(wrapped_model, steps=80, batch_size=batch_size),
        # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
        # SquareGrad(model, nb_samples=80, batch_size=batch_size),
        # VarGrad(model, nb_samples=80, batch_size=batch_size),
        # GradCAM(model),
        # GradCAMPP(model),
        # Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # Rise(model, nb_samples=500, batch_size=batch_size),
        # SobolAttributionMethod(model, batch_size=batch_size),
        # HsicAttributionMethod(wrapped_model, batch_size=batch_size),
        # Rise(wrapped_model, nb_samples=500, batch_size=batch_size),
        # Lime(model, nb_samples = 1000),
        KernelShap(wrapped_model, nb_samples = 1000, batch_size=32)
    ]
    
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
    
    total_steps = math.ceil(len(input_data) / batch)
    
    for explainer in explainers:
        # explanation methods    
        explainer_method_name = explainer.__class__.__name__
        exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
        mkdir(exp_save_path)
        
        for step in tqdm(range(total_steps), desc=explainer_method_name):
            image_names = input_data[step * batch : step * batch + batch]
            X_raw = load_and_transform_vision_data(image_names, device)

            Y_true = np.array(label[step * batch : step * batch + batch])
            labels_ohe = np.eye(class_number)[Y_true]
            
            explanations = explainer(X_raw, labels_ohe)
            if type(explanations) != np.ndarray:
                explanations = explanations.numpy()
            
            for explanation, image_name in zip(explanations, image_names):
                mkdir(exp_save_path)
                np.save(os.path.join(exp_save_path, image_name.split("/")[-1].replace(".JPEG", "")), explanation)
    
    return

main()