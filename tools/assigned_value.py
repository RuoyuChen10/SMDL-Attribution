# -*- coding: utf-8 -*-

"""
Created on 2024/8/4

@author: Ruoyu Chen
"""

import argparse

import scipy
import os
import cv2
import json
import imageio
import numpy as np
from PIL import Image

import matplotlib
from matplotlib import pyplot as plt

from scipy.ndimage import zoom

from tqdm import tqdm
from utils import *

matplotlib.get_cachedir()
plt.rc('font', family="Times New Roman")

from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Faithfulness Metric')
    parser.add_argument('--explanation-dir', 
                        type=str, 
                        default='submodular_results/cub-fair-mobilenetv2/SAM-1.0-1.0-10.0-1.0',
                        help='Save path for saliency maps generated by our methods.')
    parser.add_argument('--visualization-full', 
                        type=bool, 
                        default=True,
                        help='')
    args = parser.parse_args()
    return args

def add_value_decrease(smdl_mask, json_file):
    single_mask = np.zeros_like(smdl_mask[0].mean(-1))
    
    value_list_1 = np.array(json_file["consistency_score"]) + np.array(json_file["collaboration_score"])
    
    value_list_2 = np.array([1-json_file["collaboration_score"][-1]] + json_file["consistency_score"][:-1]) + np.array([1 - json_file["consistency_score"][-1]] + json_file["collaboration_score"][:-1])
    
    value_list = value_list_1 - value_list_2
    
    values = []
    value = 0
    for smdl_single_mask, smdl_value in zip(smdl_mask, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask.sum(-1)>0] = value
        values.append(value)
    
    attribution_map = single_mask - single_mask.min()
    attribution_map /= attribution_map.max()
    
    return attribution_map, np.array(values)

def gen_cam(image, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image cv2.COLORMAP_COOL cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_COOL)
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam, (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def visualization(image, submodular_image_set, saved_json_file, vis_image, index=None):
    image = cv2.resize(image.astype(np.uint8), (224,224))
    
    insertion_ours_images = []
    deletion_ours_images = []

    insertion_image = submodular_image_set[0] - submodular_image_set[0]
    insertion_ours_images.append(cv2.resize(insertion_image, (224,224)))
    deletion_ours_images.append(image - cv2.resize(insertion_image, (224,224)))
    for smdl_sub_mask in submodular_image_set[:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(cv2.resize(insertion_image, (224,224)))
        deletion_ours_images.append(image - cv2.resize(insertion_image, (224,224)))

    insertion_ours_images_input_results = np.array([1-saved_json_file["collaboration_score"][-1]] + saved_json_file["consistency_score"])

    if index == None:
        ours_best_index = np.argmax(insertion_ours_images_input_results)
    else:
        ours_best_index = index
    x = [(insertion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]) for insertion_ours_image in insertion_ours_images]
    i = len(x)

    fig, [ax1, ax2, ax3] = plt.subplots(1,3, gridspec_kw = {'width_ratios':[1, 1, 1.5]}, figsize=(30,8))
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title('Attribution Map', fontsize=54)
    ax1.set_facecolor('white')
    ax1.imshow(vis_image[...,::-1].astype(np.uint8))
    
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(True)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Searched Region', fontsize=54)
    ax2.set_facecolor('white')
    ax2.set_xlabel("Highest conf. {:.4f}".format(insertion_ours_images_input_results.max()), fontsize=44)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    
    ax3.set_ylabel('Recognition Score', fontsize=44)
    ax3.set_xlabel('Percentage of image revealed', fontsize=44)
    ax3.tick_params(axis='both', which='major', labelsize=36)

    x_ = x[:i]
    ours_y = insertion_ours_images_input_results[:i]
    ax3.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # draw curve
    ax3.set_facecolor('white')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['bottom'].set_linewidth(2.0)
    ax3.spines['top'].set_color('none')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(2.0)
    ax3.spines['right'].set_color('none')

    # plt.legend(["Ours"], fontsize=40, loc="upper left")
    ax3.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # Plot latest point
    # 在曲线下方填充淡蓝色
    ax3.fill_between(x_, ours_y, color='dodgerblue', alpha=0.1)

    kernel = np.ones((3, 3), dtype=np.uint8)
    # ax3.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)  # 绘制红色曲线
    ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)  # 绘制红色垂直线

    # Ours
    mask = (image - insertion_ours_images[ours_best_index]).mean(-1)
    mask[mask>0] = 1

    if ours_best_index != 0:
        dilate = cv2.dilate(mask, kernel, 3)
        # erosion = cv2.erode(dilate, kernel, iterations=3)
        # dilate = cv2.dilate(erosion, kernel, 2)
        edge = dilate - mask
        # erosion = cv2.erode(dilate, kernel, iterations=1)

    image_debug = image.copy()

    image_debug[mask>0] = image_debug[mask>0] * 0.3
    if ours_best_index != 0:
        image_debug[edge>0] = np.array([0,0,255])
    ax2.imshow(image_debug[...,::-1])
    
    auc = metrics.auc(x, insertion_ours_images_input_results)
    ax3.set_title('Insertion {:.4f}'.format(auc), fontsize=54)

def main(args):
    print(args.explanation_dir)
    
    class_ids = os.listdir(os.path.join(args.explanation_dir, "npy"))

    json_root_file = os.path.join(args.explanation_dir, "json")
    npy_root_file = os.path.join(args.explanation_dir, "npy")
    
    attribution_map_path = os.path.join(args.explanation_dir, "attribution_map")
    mkdir(attribution_map_path)
    
    visualization_path = os.path.join(args.explanation_dir, "visualization")
    mkdir(visualization_path)
    
    if args.visualization_full:
        full_visualization_path = os.path.join(args.explanation_dir, "full_visualization")
        mkdir(full_visualization_path)
    
    for class_id in tqdm(class_ids):
        json_id_files_path = os.path.join(json_root_file, class_id)
        npy_id_files_path = os.path.join(npy_root_file, class_id)
        
        # attr_id_files_path = os.path.join(attribution_map_path, class_id)
        # mkdir(attr_id_files_path)
        
        vis_id_files_path = os.path.join(visualization_path, class_id)
        mkdir(vis_id_files_path)
        
        if args.visualization_full:
            full_vis_id_files_path = os.path.join(full_visualization_path, class_id)
            mkdir(full_vis_id_files_path)

        json_file_names = os.listdir(json_id_files_path)
        for json_file_name in json_file_names:
            json_file_path = os.path.join(json_id_files_path, json_file_name)
            npy_file_path = os.path.join(npy_id_files_path, json_file_name.replace(".json", ".npy"))
            
            save_attribution_map_path = os.path.join(attribution_map_path, json_file_name.replace(".json", ".npy"))
            save_visualization_map_path = os.path.join(vis_id_files_path, json_file_name.replace(".json", ".png"))
            
            if args.visualization_full:
                save_full_visualization_map_path = os.path.join(full_vis_id_files_path, json_file_name.replace(".json", ".png"))
                
                if os.path.exists(save_full_visualization_map_path):
                    continue

            with open(json_file_path, 'r', encoding='utf-8') as f:
                saved_json_file = json.load(f)            
            submodular_image_set = np.load(npy_file_path)
            
            image = submodular_image_set.sum(0)
    
            attribution_map, value_list = add_value_decrease(submodular_image_set, saved_json_file)

            im, heatmap = gen_cam(image, norm_image(attribution_map))
            
            np.save(save_attribution_map_path, attribution_map)
            cv2.imwrite(save_visualization_map_path, im.astype(np.uint8))

            if args.visualization_full:
                try:
                    im = cv2.resize(im, (224,224))
                    visualization(image, submodular_image_set, saved_json_file, im)
                    plt.savefig(save_full_visualization_map_path, bbox_inches='tight',pad_inches=0.0)
                    plt.clf()
                    plt.close()
                except:
                    pass

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
