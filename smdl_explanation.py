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

import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *

red_tr    = get_alpha_cmap('Reds')

from models.submodular import FaceSubModularExplanation

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
    parser.add_argument('--partition',
                        type=str,
                        default="grad",
                        choices=["grad", "pixel"],
                        help="")
    parser.add_argument('--random-patch',
                        type=bool,
                        default=False,
                        help="")
    parser.add_argument('--explanation-method', 
                        type=str, 
                        default='./explanation_results/celeba/HsicAttributionMethod',
                        help='Save path for saliency maps generated by interpretability methods.')
    parser.add_argument('--sub-n', 
                        type=int, default=2,
                        help='')
    parser.add_argument('--sub-k', 
                        type=int, default=40,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/celeba',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def Partition_image(image, explanation_mask, partition_number=112):

    b,g,r = cv2.split(image)

    explanation_mask_flatten = explanation_mask.flatten()
    
    index = np.argsort(-explanation_mask_flatten)
    
    pixels_per_partition = int(len(explanation_mask_flatten) / partition_number)
    
    components_image_list = []
    components_index_list = []
    for i in range(partition_number):
        b_tmp = b.flatten()
        g_tmp = g.flatten()
        r_tmp = r.flatten()
        
        cp_index = np.zeros_like(b_tmp)
        cp_index[index[i*pixels_per_partition: (i+1)*pixels_per_partition]] = 1
        
        b_tmp[index[ : i*pixels_per_partition]] = 0
        g_tmp[index[ : i*pixels_per_partition]] = 0
        r_tmp[index[ : i*pixels_per_partition]] = 0

        b_tmp[index[(i+1)*pixels_per_partition :]] = 0
        g_tmp[index[(i+1)*pixels_per_partition :]] = 0
        r_tmp[index[(i+1)*pixels_per_partition :]] = 0

        b_tmp = b_tmp.reshape((image.shape[0], image.shape[1]))
        g_tmp = g_tmp.reshape((image.shape[0], image.shape[1]))
        r_tmp = r_tmp.reshape((image.shape[0], image.shape[1]))
        cp_index = cp_index.reshape((image.shape[0], image.shape[1]))
        
        img_tmp = cv2.merge([b_tmp, g_tmp, r_tmp])
        components_image_list.append(img_tmp)
        components_index_list.append(cp_index)
    return components_image_list#, components_index_list

def partition_by_mulit_grad(image, explanation_mask, grad_size = 28, grad_num_per_set = 8):
    """
    Divide the image into grad_size x grad_size areas, divide according to eplanation_mask, each division has grad_num_per_set grads.
    """
    partition_number = int(grad_size * grad_size / grad_num_per_set)
    # pixel_length_per_grad = int(image.shape[0] / grad_size)

    components_image_list = []
    pool_z = cv2.resize(explanation_mask, (grad_size, grad_size))

    pool_z_flatten = pool_z.flatten()
    index = np.argsort(- pool_z_flatten)     # From high to low

    for i in range(partition_number):
        binary_mask = np.zeros_like(index)
        binary_mask[index[i*grad_num_per_set : (i+1)*grad_num_per_set]] = 1
        binary_mask = binary_mask.reshape((grad_size, grad_size, 1))
        binary_mask = cv2.resize(
            binary_mask, (image.shape[0],image.shape[1]), interpolation=cv2.INTER_NEAREST)

        components_image_list.append(
            (image * binary_mask[:, :, np.newaxis]).astype(np.uint8)
        )
    return components_image_list

def Partition_by_patch(image, partition_size=10):
    pixel_length = int(image.shape[0] / partition_size)
    
    components_image_list = []
    for i in range(partition_size):
        for j in range(partition_size):
            image_tmp = np.zeros_like(image)
            image_tmp[i*pixel_length : (i+1)*pixel_length, j*pixel_length : (j+1)*pixel_length] = image[i*pixel_length : (i+1)*pixel_length, j*pixel_length : (j+1)*pixel_length]
            
            components_image_list.append(image_tmp)
    return components_image_list

def main(args):
    
    smdl = FaceSubModularExplanation(n=args.sub_n, k=args.sub_k)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    mkdir(args.save_dir)
    if args.random_patch:
        save_dir = os.path.join(args.save_dir, "random_patch" + "-" + str(args.sub_k))
    else:
        save_dir = os.path.join(args.save_dir, args.explanation_method.split("/")[-1] + "-" + str(args.sub_k))
    mkdir(save_dir)
    
    for info in tqdm(infos):
        id_people = info.split(" ")[-1]
        # save_people_path = os.path.join(save_dir, id_people)
        # mkdir(save_people_path)
        
        image_relative_path = info.split(" ")[0]
        
        if not args.random_patch:
            mask_path = os.path.join(args.explanation_method, image_relative_path.replace(".jpg", ".npy"))
            
            mask = np.load(mask_path)
        
        # Ground Truth Label
        # gt_label = int(id_people)
        
        # Read original image
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        
        if args.random_patch:
            components_image_list = Partition_by_patch(image)
        else:
            if args.partition == "pixel":
                components_image_list = Partition_image(image, mask)
            elif args.partition == "grad":
                components_image_list = partition_by_mulit_grad(image, mask)

        submodular_image, submodular_image_set, saved_json_file = smdl(components_image_list)
        
        # Save the final image
        save_image_root_path = os.path.join(save_dir, "image-{}".format(args.sub_k))
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
        save_josn_root_path = os.path.join(save_dir, "json")
        mkdir(save_josn_root_path)
        mkdir(os.path.join(save_josn_root_path, id_people))
        with open(os.path.join(save_josn_root_path, image_relative_path.replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))


if __name__ == "__main__":
    args = parse_args()
    main(args)