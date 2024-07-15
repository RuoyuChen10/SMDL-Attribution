# -*- coding: utf-8 -*-

"""
Created on 2024/6/11

@author: Ruoyu Chen
LanguageBind ViT version
Efficient Submodular Attribution
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

# https://github.com/facebookresearch/ImageBind
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

import torch
from torchvision import transforms

red_tr = get_alpha_cmap('Reds')

from models.submodular_vit_efficient import MultiModalSubModularExplanationEfficientV2

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_languagebind_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
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
                        type=str, default='./submodular_results/imagenet-languagebind-efficientv2/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,T,W,H]) # video
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "video": {'pixel_values': vision_inputs},
        }
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
        
        return embeddings["video"]

def transform_languagebind_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,t,w,h]
    """
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
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = data_transform(image)
    image = image.unsqueeze(1).repeat(1, 8, 1, 1)
    return image

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = to_device(tokenizer(texts, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device) #tokenize
            input = {
                "language": texts
            }
            with torch.no_grad():
                class_embeddings = model(input)["language"]

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).cuda()
    return zeroshot_weights

def main(args):
    
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # Instantiate model
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'thermal': 'LanguageBind_Thermal',
        'image': 'LanguageBind_Image',
        'depth': 'LanguageBind_Depth',
    }
    model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    model = model.to(device)
    model.eval()

    vis_model = LanguageBindModel_Super(model)
    print("load languagebind model")
    
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')

    
    semantic_path = "ckpt/semantic_features/languagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    else:
        semantic_feature = zeroshot_classifier(model, imagenet_classes, imagenet_templates, tokenizer, device)
        torch.save(semantic_feature, semantic_path)
    
    
    smdl = MultiModalSubModularExplanationEfficientV2(
        vis_model, semantic_feature, transform_languagebind_vision_data, device=device, 
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
    
    select_infos = infos[args.begin : args.end]
    for info in tqdm(select_infos):
        gt_id = info.split(" ")[1]
        
        image_relative_path = info.split(" ")[0]
        
        if os.path.exists(
            os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".JPEG", ".json"))
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
        
        # Save the final image
        # save_image_root_path = os.path.join(save_dir, "image")
        # mkdir(save_image_root_path)
        # mkdir(os.path.join(save_image_root_path, gt_id))
        # save_image_path = os.path.join(
        #     save_image_root_path, image_relative_path)
        # cv2.imwrite(save_image_path, submodular_image)

        # Save npy file
        mkdir(os.path.join(save_npy_root_path, gt_id))
        np.save(
            os.path.join(
                os.path.join(save_npy_root_path, gt_id), image_relative_path.replace(".JPEG", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        mkdir(os.path.join(save_json_root_path, gt_id))
        with open(os.path.join(
            os.path.join(save_json_root_path, gt_id), image_relative_path.replace(".JPEG", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

    #     # Save GIF
    #     save_gif_root_path = os.path.join(save_dir, "gif")
    #     mkdir(save_gif_root_path)
    #     save_gif_path = os.path.join(save_gif_root_path, gt_id)
    #     mkdir(save_gif_path)

        # img_frame = submodular_image_set[0][..., ::-1]
        # frames = []
        # frames.append(img_frame)
        # for fps in range(1, submodular_image_set.shape[0]):
        #     img_frame = img_frame.copy() + submodular_image_set[fps][..., ::-1]
        #     frames.append(img_frame)

        # imageio.mimsave(os.path.join(save_gif_root_path, image_relative_path.replace(".jpg", ".gif")), 
        #                       frames, 'GIF', duration=0.0085)  


if __name__ == "__main__":
    args = parse_args()
    
    main(args)