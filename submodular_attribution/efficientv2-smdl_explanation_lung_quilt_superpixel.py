# -*- coding: utf-8 -*-

"""
Created on 2024/6/17

@author: Ruoyu Chen
CLIP ViT version
"""

import argparse

import scipy
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image

import subprocess
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

from open_clip import create_model_from_pretrained, get_tokenizer

import torch
from torchvision import transforms

red_tr = get_alpha_cmap('Reds')

from models.submodular_vit_efficient import MultiModalSubModularExplanationEfficientV2

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

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/medical_lung/lung_dataset',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/medical_lung/LC25000_lung_quilt_1k_false.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=0.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=0.05,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda4', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--pending-samples',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/lung-quilt-efficientv2-debug/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class QuiltModel_Super(torch.nn.Module):
    def __init__(self, 
                 download_root=".checkpoints/QuiltNet-B-32",
                 device = "cuda"):
        super().__init__()
        self.model, _ = create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32', cache_dir=download_root)
        self.device = device
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image = data_transform(image)
    return image

def main(args):
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate model
    vis_model = QuiltModel_Super()
    vis_model.eval()
    vis_model.to(device)
    print("load Quilt-1M model")
    
    tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    texts = tokenizer([lc_lung_template + l for l in lc_lung_classes], context_length=77).to(device)

    with torch.no_grad():
        semantic_feature = vis_model.model.encode_text(texts) * 10
    
    smdl = MultiModalSubModularExplanationEfficientV2(
        vis_model, semantic_feature, transform_vision_data, device=device, 
        lambda1=args.lambda1, 
        lambda2=args.lambda2, 
        lambda3=args.lambda3, 
        lambda4=args.lambda4,
        pending_samples=args.pending_samples)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-{}-{}-pending-samples-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.lambda4, args.pending_samples))  
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_infos = infos[args.begin : end]
    for info in tqdm(select_infos):
        gt_id = info.split(" ")[1]
        
        image_relative_path = info.split(" ")[0]
        
        if os.path.exists(
            os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".jpeg", ".json"))
        ):
            continue
        
        # Ground Truth Label
        gt_label = int(gt_id)
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
        smdl.k = len(element_sets_V)

    #     start = time.time()
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, gt_label)
    #     end = time.time()
    #     # print('程序执行时间: ',end - start)

        # Save npy file
        mkdir(os.path.join(save_npy_root_path, gt_id))
        np.save(
            os.path.join(
                os.path.join(save_npy_root_path, gt_id), image_relative_path.replace(".jpeg", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        mkdir(os.path.join(save_json_root_path, gt_id))
        with open(os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".jpeg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    args = parse_args()
    
    main(args)