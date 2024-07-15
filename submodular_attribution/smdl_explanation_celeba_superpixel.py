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
import time

import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *

red_tr    = get_alpha_cmap('Reds')

from models.submodular_face import FaceSubModularExplanation

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/celeb-a/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/celeb-a/eval.txt',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--sub-n', 
                        type=int, default=1,
                        help='')
    parser.add_argument('--sub-k', 
                        type=int, default=97,
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
                        default="configs/celeba/submodular_cfg_celeba_tf.json",
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/celeba',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=15, ruler = 40.0) 
        slic.iterate(20)     # The number of iterations, the larger the better the effect
        label_slic = slic.getLabels()        # Get superpixel label
        number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

        for i in range(number_slic):
            img_copp = image.copy()
            img_copp = img_copp * (label_slic == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image,10)  # The input image size must be the same as the initialization shape and the number of iterations is 10
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = image.copy()
            img_copp = img_copp * (label_seeds == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    return element_sets_V

def main(args):
    
    smdl = FaceSubModularExplanation(
        cfg_path=args.cfg, n=args.sub_n, k=args.sub_k, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3, lambda4=args.lambda4)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    
    save_dir = os.path.join(args.save_dir, "superpixel-{}-{}-{}-{}-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.lambda4))
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    save_josn_root_path = os.path.join(save_dir, "json")
    mkdir(save_josn_root_path)
    for info in tqdm(infos):
        id_people = info.split(" ")[-1]
        # save_people_path = os.path.join(save_dir, id_people)
        # mkdir(save_people_path)
        
        image_relative_path = info.split(" ")[0]
        
        if os.path.exists(
            os.path.join(save_josn_root_path, image_relative_path.replace(".jpg", ".json"))
        ):
            continue
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
        smdl.k = len(element_sets_V)
        
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V, int(id_people))
        
        # Save the final image
        save_image_root_path = os.path.join(save_dir, "image-{}".format(args.sub_k))
        mkdir(save_image_root_path)
        mkdir(os.path.join(save_image_root_path, id_people))
        save_image_path = os.path.join(
            save_image_root_path, image_relative_path)
        cv2.imwrite(save_image_path, submodular_image)

        # Save npy file
        mkdir(os.path.join(save_npy_root_path, id_people))
        np.save(
            os.path.join(save_npy_root_path, image_relative_path.replace(".jpg", ".npy")),
            np.array(submodular_image_set)
        )

        # Save json file
        mkdir(os.path.join(save_josn_root_path, id_people))
        with open(os.path.join(save_josn_root_path, image_relative_path.replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))


if __name__ == "__main__":
    args = parse_args()
    main(args)