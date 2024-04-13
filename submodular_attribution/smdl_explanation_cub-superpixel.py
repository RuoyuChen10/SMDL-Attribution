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
    parser.add_argument('--division',
                        type=str,
                        default="superpixel",
                        choices=["grad", "pixel", "superpixel"],
                        help="")
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--sub-n', 
                        type=int, default=1,
                        help='')
    parser.add_argument('--sub-k', 
                        type=int, default=24,
                        help='')
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

def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler = 20.0) 
        slic.iterate(20)     #迭代次数，越大效果越好
        label_slic = slic.getLabels()        #获取超像素标签
        number_slic = slic.getNumberOfSuperpixels()  #获取超像素数目

        for i in range(number_slic):
            img_copp = image.copy()
            img_copp = img_copp * (label_slic == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image,10)  #输入图像大小必须与初始化形状相同，迭代次数为10
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = image.copy()
            img_copp = img_copp * (label_seeds == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    return element_sets_V

def main(args):
    
    smdl = CubSubModularExplanationV2(cfg_path=args.cfg, k=args.sub_k, 
                                    lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3, lambda4=args.lambda4)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    if args.division == "superpixel":
        save_dir = os.path.join(args.save_dir, args.division + "-{}-{}-{}-{}-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.lambda4))  
    
    mkdir(save_dir)
    
    for info in tqdm(infos[:]):
        id_people = info.split(" ")[-1]
        # save_people_path = os.path.join(save_dir, id_people)
        # mkdir(save_people_path)
        
        image_relative_path = info.split(" ")[0]
        
        # Ground Truth Label
        # gt_label = int(id_people)
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
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
        save_gif_root_path = os.path.join(save_dir, "gif")
        mkdir(save_gif_root_path)
        save_gif_path = os.path.join(save_gif_root_path, id_people)
        mkdir(save_gif_path)

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