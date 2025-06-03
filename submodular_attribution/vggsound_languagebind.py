# -*- coding: utf-8 -*-

"""
Created on 2024/8/22

@author: Ruoyu Chen
Languagebind audio ViT version
"""

import argparse

import scipy
import os
import cv2
import json
import numpy as np
from PIL import Image

import subprocess
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib import pyplot as plt
# plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import torch
from torchvision import transforms

red_tr = get_alpha_cmap('Reds')

from models.submodular_audio_efficient_plus import AudioSubModularExplanationEfficientPlus

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/vggsound/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/vggsound/val_languagebind_600_true.txt',
                        help='Datasets.')
    parser.add_argument('--lambda1', 
                        type=float, default=0.01,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=0.05,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=20.,
                        help='')
    parser.add_argument('--lambda4', 
                        type=float, default=5.,
                        help='')
    parser.add_argument('--pending-samples',
                        type=int,
                        default=20,
                        help='')
    parser.add_argument('--grad-partition-size',
                        type=int,
                        default=10,
                        help="")
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/vggsound-languagebind-efficientv2/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, audio_inputs):
        """
        Input:
            audio_inputs: torch.size([B,C,W,H]) # video
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "audio": {'pixel_values': audio_inputs},
        }
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
        
        return embeddings["audio"]

def Partition_by_patch(image, partition_size=(8,8)):
    """_summary_

    Args:
        image (torch): [c,w,h]
        partition_size (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    pixel_length = image.shape[1] / partition_size[0]
    pixel_width = image.shape[2] / partition_size[1]
    
    components_image_list = []
    for i in range(partition_size[0]):
        for j in range(partition_size[1]):
            image_tmp = np.zeros_like(image)
            image_tmp[:,int(i*pixel_length) : int((i+1)*pixel_length), int(j*pixel_width) : int((j+1)*pixel_width)] = image[:, int(i*pixel_length) : int((i+1)*pixel_length), int(j*pixel_width) : int((j+1)*pixel_width)]
            
            components_image_list.append(image_tmp)
    return components_image_list

def read_audio(
    audio_path,
    modality_transform,
    device = "cpu"
):
    audio = [audio_path]
    audio_proccess = to_device(modality_transform['audio'](audio), device)['pixel_values'][0]
    return audio_proccess.cpu().numpy()

def transform_audio_data(audio_numpy):
    audio = torch.from_numpy(audio_numpy)
    return audio

def main(args):
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # Instantiate model
    clip_type = {
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
    }
    model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    model.eval()
    model.to(device)
    
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
    
    audio_model = LanguageBindModel_Super(model)
    print("load languagebind model")
    
    semantic_path = "ckpt/semantic_features/vggsound_languagebind_cls.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
    
    else:
        text = [vggsound_template.format(vggsound_class) for vggsound_class in vggsound_classes]
        pretrained_ckpt = f'lb203/LanguageBind_Image'
        tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')
        inputs = {}
        inputs['language'] = to_device(tokenizer(text, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)
        with torch.no_grad():
            embeddings = model(inputs)
        semantic_feature = embeddings['language']
        torch.save(embeddings['language'], semantic_path)
    
    smdl = AudioSubModularExplanationEfficientPlus(
        audio_model, semantic_feature, transform_audio_data, device=device, 
        lambda1=args.lambda1, 
        lambda2=args.lambda2, 
        lambda3=args.lambda3, 
        lambda4=args.lambda4,
        pending_samples=args.pending_samples)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "sound-{}-{}-{}-{}-pending-samples-{}-divison-{}".format(args.lambda1, args.lambda2, args.lambda3, args.lambda4, args.pending_samples, args.grad_partition_size))  
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    select_infos = infos[args.begin : args.end]
    for info in tqdm(select_infos):
        gt_id = info.split(" ")[1]
        
        audio_relative_path = info.split(" ")[0]
        
        if os.path.exists(
            os.path.join(
            os.path.join(save_json_root_path, gt_id), audio_relative_path.replace(".flac", ".json"))
        ):
            continue
        
        # Ground Truth Label
        gt_label = int(gt_id)
        
        # Read original audio
        audio_path = os.path.join(args.Datasets, audio_relative_path)
        audio_input = read_audio(audio_path, modality_transform, device)
        
        element_sets_V = Partition_by_patch(audio_input, (args.grad_partition_size, args.grad_partition_size))
        smdl.k = len(element_sets_V)

        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, gt_label)

        # Save npy file
        mkdir(os.path.join(save_npy_root_path, gt_id))
        np.save(
            os.path.join(
                os.path.join(save_npy_root_path, gt_id), audio_relative_path.replace(".flac", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        mkdir(os.path.join(save_json_root_path, gt_id))
        with open(os.path.join(
            os.path.join(save_json_root_path, gt_id), audio_relative_path.replace(".flac", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))


if __name__ == "__main__":
    args = parse_args()
    
    main(args)