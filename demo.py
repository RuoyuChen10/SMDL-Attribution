import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.applications.resnet import (
    preprocess_input)

import imageio

method = "GradCAMPP"
image_path = "datasets/CUB/test/139/Summer_Tanager_0101_139441.jpg"
hsic_mask_path = "explanation_results/cub/{}/139/Summer_Tanager_0101_139441.npy".format(method)
ours_mask_path = "submodular_results/cub/grad-10x10-2/{}-49-1.0-1.0-1.0-1.0/npy/139/Summer_Tanager_0101_139441.npy".format(method)
class_index = 139
steps = 50

keras_model_path = "ckpt/keras_model/cub-resnet101.h5"
model = load_model(keras_model_path)

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

def main():
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    explanation = np.load(hsic_mask_path)
    smdl_mask = np.load(ours_mask_path)
    
    insertion_explanation_images = []
    deletion_explanation_images = []

    for i in range(1, steps+1):
        perturbed_rate = i / steps
        insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
        deletion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "deletion"))
    
    
    insertion_ours_images = []
    deletion_ours_images = []

    insertion_image = smdl_mask[0]
    insertion_ours_images.append(insertion_image)
    deletion_ours_images.append(image - insertion_image)
    for smdl_sub_mask in smdl_mask[1:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(insertion_image)
        deletion_ours_images.append(image - insertion_image)
    insertion_ours_images.append(image)
    deletion_ours_images.append(image - image)
    
    insertion_explanation_images_input = preprocess_input(
        np.array(insertion_explanation_images)[..., ::-1]
    )
    deletion_explanation_images_input = preprocess_input(
        np.array(deletion_explanation_images)[..., ::-1]
    )

    insertion_ours_images_input = preprocess_input(
        np.array(insertion_ours_images)[..., ::-1]
    )
    deletion_ours_images_input = preprocess_input(
        np.array(deletion_ours_images)[..., ::-1]
    )
    
    insertion_explanation_images_input_results = model(insertion_explanation_images_input)[:,class_index]

    insertion_ours_images_input_results = model(insertion_ours_images_input)[:,class_index]
        
    frames = []
    x = list(np.linspace(0,1,steps))
    for i in range(len(x)):
        fig, [ax1, ax2, ax3] = plt.subplots(1,3, gridspec_kw = {'width_ratios':[1, 1, 1.5]}, figsize=(30,8))
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.imshow(insertion_explanation_images[i][...,::-1])
        ax1.set_title(method, fontsize=32)

        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.imshow(insertion_ours_images[i][...,::-1])
        ax2.set_title('Ours', fontsize=32)

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.title('Insertion', fontsize=32)
        plt.ylabel('Recognition Score', fontsize=28)

        x_ = x[:i+1]
        explanation_y = insertion_explanation_images_input_results.numpy()[:i+1]
        plt.plot(x_, explanation_y, color='orange', linewidth=3.5)  # 绘制曲线

        ours_y = insertion_ours_images_input_results.numpy()[:i+1]
        plt.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # 绘制曲线

        plt.legend(['Hsic-Attribution', "+ Ours"], fontsize=24, loc="upper left")
        plt.scatter(x_[-1], explanation_y[-1], color='orange', s=54)  # 绘制最新点
        plt.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # 绘制最新点
        
        plt.savefig("gif_tmp.png", bbox_inches='tight')
        img_frame = cv2.imread("gif_tmp.png")
        frames.append(img_frame.copy()[...,::-1])
    
    for j in range(10):
        frames.append(img_frame.copy()[...,::-1])
    
    imageio.mimsave(image_path.split("/")[-1].replace(".jpg", ".gif"), frames, 'GIF', duration=0.1)  

main()