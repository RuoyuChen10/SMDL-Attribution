# -*- coding: utf-8 -*-  

"""
Created on 2023/5/13

@author: Ruoyu Chen
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import json
from itertools import combinations
import math

from mtcnn.src import detect_faces
from utils import *
from models.submodular import SubModular

import torchvision.transforms.functional as TF
import torch.nn.functional as F

mt = "VGGFace2"

if mt == "VGGFace2":
    ID_results_path = "motivation/results/VGGFace2/ID"
    ID_image_path = "motivation/images/VGGFace2/ID"
    ID_names = [
        ('n000307', 290),
        ('n000309', 292),
        ('n000337', 320),
        ('n000353', 336),
        ('n000359', 342),
        ('n003021', 2816),
        ('n003197', 2984),
        ('n005546', 5144),
        ('n006579', 6103),
        ('n006634', 6156),
    ]
    id_person2id = {
        'n000307': 290,
        'n000309': 292,
        'n000337': 320,
        'n000353': 336,
        'n000359': 342,
        'n003021': 2816,
        'n003197': 2984,
        'n005546': 5144,
        'n006579': 6103,
        'n006634': 6156
    }
    
    # ID model
    model_path = "ckpt/ArcFace-VGGFace2-R50-8631.onnx"
    class_num = 8631
    
    # submodular parameters
    batch_size = 400    # RTX 3090: 450
    
    combination_number_k = 5

    transforms = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    vis_topk = 20
    interval = 0.025
    
def prepare_image(path, size=112):
    img = Image.open(path)
    img = transforms(img).numpy()
    # img = img.transpose(1, 2, 0)
    
    return img

def convert_prepare_image(image, size=112):
    image = cv2.resize(image, (size, size))
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = transforms(img).numpy()
    # img = img.transpose(1, 2, 0)
    return img

def combination_mask(image_list, k = None):
    """permutation
    :image_list: [image1, image2, ...]
    """
    org_img_num = len(image_list)
    index = list(range(org_img_num))
    
    if k == None:
        k = org_img_num + 1
    else:
        k = k + 1
    
    combination_list = []
    
    # number for combination
    for comb_num in range(2, k):
        # Combination
        sub_index_list = list(combinations(index, comb_num))
        for sub_index in sub_index_list:
            # combinate the masked images
            combination_mask_image = np.zeros_like(image_list[0])
            for idx in sub_index:
                combination_mask_image = combination_mask_image + image_list[idx]
            image_list.append(combination_mask_image)
            combination_list.append(sub_index)

    return image_list, combination_list

def softmax(f, tau=1):
    # f -= np.max(f)
    return np.exp(f / tau) / np.exp(f / tau).sum(axis=1, keepdims=True)

def main():
    # Load SMDL model, default parameters: models/submodular_cfg.json
    smdl = SubModular(combination_number_k = combination_number_k)
    
    id_peoples = os.listdir(ID_image_path)
    for id_person in id_peoples:
        if ".py" in id_person:
            continue
    
        gt_label = id_person2id[id_person]
        
        id_person_path = os.path.join(ID_results_path, id_person)
        
        # load image names
        image_txt = os.path.join(id_person_path, "image.txt")
        with open(image_txt, "r") as f:
            image_names = f.read().split('\n')
            while "" in image_names:
                image_names.remove("")
                
        # load masks
        explanation_masks = np.load(
            os.path.join(id_person_path, "explanation.npy")
        )
        
        # Loop image names
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(os.path.join(ID_image_path, id_person), image_name)
            print(image_path)
            image = cv2.imread(image_path)
            
            mask = norm(explanation_masks[i])[:, :, np.newaxis]

            mask_images = []
            components_image_list = []
            
            for erasing_threshold in (np.arange(interval, 1, interval) + interval):
                masked_image = image * (mask < erasing_threshold).astype(int) * (mask > erasing_threshold-interval).astype(int)
                mask_images.append(masked_image.astype(np.uint8))
                
                components_image_list.append(masked_image.astype(np.uint8))
            
            # source image
            source_image = np.array([convert_prepare_image(image)])

            smdl_score, u_, r_, mc_, fr_r, combination_list = smdl(components_image_list, source_image, convert_prepare_image, batch_size)
            
            predicted_ids = np.argmax(fr_r, axis=1) == gt_label # if face predict model is true
            predicted_score = softmax(fr_r)[:, gt_label]
            
            inds = np.argsort(smdl_score)[-vis_topk:]
            
            top_combination_list = []
            for ind in inds:
                top_combination_list.append(combination_list[ind])
                
            image_list = smdl._combinate_sample(components_image_list, top_combination_list)
            
            # visualization
            plt.figure(figsize=(32,32))
            for j, ind in enumerate(inds):
                ax = plt.subplot(1, vis_topk, vis_topk-j)
            
                plt.axis('off')
                plt.imshow(Image.fromarray(cv2.cvtColor(image_list[j], cv2.COLOR_BGR2RGB)))

                # mtcnn_detect
                bx, ldmrk = detect_faces(Image.fromarray(cv2.cvtColor(image_list[j], cv2.COLOR_BGR2RGB)), thresholds = [0.6, 0.7, 0.85])

                ax.set_title(
                    "{:.5f}\n{:.5f}\n{:.5f}\n{:.5f}\n{:.5f}\n{}\n{}\n{}".format(
                        smdl_score[ind], u_[ind], r_[ind], mc_[ind], 
                        predicted_score[ind], predicted_ids[ind], len(bx), str(combination_list[ind])),
                    y=-1.9,
                    fontsize=16)
                if j == vis_topk:
                    break
            plt.savefig("results/{}.jpg".format(id_person), bbox_inches='tight')
            
            break
        
    return 


if __name__ == '__main__':
    main()
