# -*- coding: utf-8 -*-

"""
Created on 2023/7/26

@author: Ruoyu Chen
V2 version
"""

import argparse

import scipy
import os
import numpy as np
import cv2
import json
import imageio

import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

red_tr    = get_alpha_cmap('Reds')

from models.submodular_cub_v2 import CubSubModularExplanationV2

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/CUB/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/CUB/eval_random.txt',
                        help='Datasets.')
    parser.add_argument('--sam-mask-dir',
                        type=str,
                        default="SAM_mask/CUB-resnet",
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
    parser.add_argument('--cfg', 
                        type=str, 
                        default="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/cub-resnet',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def main(args):
    
    smdl = CubSubModularExplanationV2(cfg_path=args.cfg, 
                                    lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3, lambda4=args.lambda4)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "SAM-{}-{}-{}-{}".format(args.lambda1, args.lambda2, args.lambda3, args.lambda4))  
    
    mkdir(save_dir)
    
    for info in tqdm(infos[:]):
        id_people = info.split(" ")[-1]
        
        image_relative_path = info.split(" ")[0]
        
        element_sets_V = np.load(os.path.join(args.sam_mask_dir, image_relative_path.replace(".jpg", ".npy")))
        smdl.k = len(element_sets_V)

        start = time.time()
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V)
        end = time.time()
        # print('程序执行时间: ',end - start)
        
        # Save the final image
        save_image_root_path = os.path.join(save_dir, "image")
        mkdir(save_image_root_path)
        mkdir(os.path.join(save_image_root_path, id_people))
        save_image_path = os.path.join(
            save_image_root_path, image_relative_path)
        cv2.imwrite(save_image_path, submodular_image)

        # Save npy file
        save_npy_root_path = os.path.join(save_dir, "npy")
        mkdir(save_npy_root_path)
        mkdir(os.path.join(save_npy_root_path, id_people))
        np.save(
            os.path.join(save_npy_root_path, image_relative_path.replace(".jpg", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        save_json_root_path = os.path.join(save_dir, "json")
        mkdir(save_json_root_path)
        mkdir(os.path.join(save_json_root_path, id_people))
        with open(os.path.join(save_json_root_path, image_relative_path.replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

        # Save GIF
        # save_gif_root_path = os.path.join(save_dir, "gif")
        # mkdir(save_gif_root_path)
        # save_gif_path = os.path.join(save_gif_root_path, id_people)
        # mkdir(save_gif_path)

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