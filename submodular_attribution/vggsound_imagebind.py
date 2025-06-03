# -*- coding: utf-8 -*-

"""
Created on 2024/8/22

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
# plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

# https://github.com/facebookresearch/ImageBind
from imagebind import data
from imagebind.data import waveform2melspec, get_clip_timepoints
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import torch
from torchvision import transforms

import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

red_tr = get_alpha_cmap('Reds')

from models.submodular_audio_efficient_plus import AudioSubModularExplanationEfficientPlus

clip_sampler = ConstantClipsPerVideoSampler(
    clip_duration=2, clips_per_video=3
)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for ImageBind Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/vggsound/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/vggsound/val_imagebind_600_true.txt',
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
                        default=12,
                        help='')
    parser.add_argument('--grad-partition-size',
                        type=int,
                        default=8,
                        help="")
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/vggsound-imagebind-efficientv2/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, audio_inputs):
        """
        Input:
            audio_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "audio": audio_inputs,
        }
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
        
        return embeddings["audio"]

def Partition_by_patch(image, partition_size=(8,8)):
    """_summary_

    Args:
        image (torch): [c,d,w,h]
        partition_size (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    pixel_length = image.shape[2] / partition_size[0]
    pixel_width = image.shape[3] / partition_size[1]
    
    components_image_list = []
    for i in range(partition_size[0]):
        for j in range(partition_size[1]):
            image_tmp = np.zeros_like(image)
            image_tmp[:,:,int(i*pixel_length) : int((i+1)*pixel_length), int(j*pixel_width) : int((j+1)*pixel_width)] = image[:,:, int(i*pixel_length) : int((i+1)*pixel_length), int(j*pixel_width) : int((j+1)*pixel_width)]
            
            components_image_list.append(image_tmp)
    return components_image_list

def read_audio(
    audio_path,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    mean= -4.268, 
    std= 9.138
):
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    return all_clips.cpu().numpy()

def transform_audio_data(audio_numpy):
    audio = torch.from_numpy(audio_numpy)
    return audio

def main(args):
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
   # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    audio_model = ImageBindModel_Super(model)
    print("load imagebind model")
    
    semantic_path = "ckpt/semantic_features/vggsound_imagebind_cls.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device) * 0.05
    
    else:
        text = [vggsound_template.format(vggsound_class) for vggsound_class in vggsound_classes]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text , device),
        }
        with torch.no_grad():
            embeddings = model(inputs)
        semantic_feature = embeddings[ModalityType.TEXT] * 0.05
        torch.save(embeddings[ModalityType.TEXT], semantic_path)
    
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
        audio_input = read_audio(audio_path, device)
        
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