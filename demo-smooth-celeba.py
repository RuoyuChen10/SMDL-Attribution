import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.applications.resnet import (
    preprocess_input)
from insight_face_models import *
import imageio
import tensorflow_addons as tfa

matplotlib.get_cachedir()
plt.rc('font', family="Times New Roman")
# method = "GradCAMPP"
# method_name = "GradCAM++"
method = "HsicAttributionMethod"
method_name = "HSIC-Attribution"
image_path = "datasets/celeb-a/test/7362/084204.jpg"
hsic_mask_path = "explanation_results/celeba/{}/7362/084204.npy".format(method)
ours_mask_path = "submodular_results/celeba-cross-model/grad-28x28-8/{}-97-1.0-1.0-1.0-1.0/npy/7362/084204.npy".format(method)
# ours_mask_path = "submodular_results/cub/grad-10x10-2/{}-49-1.0-1.0-1.0-1.0/npy/7362/084204.npy".format(method)
class_index = int(image_path.split("/")[-2])
steps = 98

softmax = tf.keras.layers.Softmax(axis=-1)

keras_model_path = "ckpt/keras_model/keras-ArcFace-R100-Celeb-A.h5"
model = load_model(keras_model_path)

def load_image(imgs, size=112):
    img_ = []
    for img in imgs:
        img = cv2.resize(img, (size, size))
        img = (img - 127.5) * 0.0078125
        img_.append(img.astype(np.float32))
    return np.array(img_).astype(np.float32)

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i
        
        single_mask = single_mask.mean(-1)
        single_mask = cv2.resize(single_mask.astype(np.uint8), (7,7))    # for smooth
        single_mask = cv2.resize(single_mask.astype(np.uint8), (112,112))
        # single_mask = np.exp(single_mask / single_mask.max() / 0.1)
        # single_mask = (single_mask / single_mask.max())*255
        
        batch_mask.append(single_mask)

    return np.array(batch_mask)

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
    image = cv2.resize(image, (112,112))
    explanation = np.load(hsic_mask_path)
    smdl_mask = np.load(ours_mask_path)
    smdl_mask_convert = convert_smdl_mask([smdl_mask])[0]
    
    insertion_explanation_images = []
    deletion_explanation_images = []
    insertion_ours_images = []
    deletion_ours_images = []

    for i in range(0, steps+1):
        perturbed_rate = i / steps
        insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
        deletion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "deletion"))
        insertion_ours_images.append(perturbed(image, smdl_mask_convert, rate = perturbed_rate, mode = "insertion"))
        deletion_ours_images.append(perturbed(image, smdl_mask_convert, rate = perturbed_rate, mode = "deletion"))

    # insertion_image = smdl_mask[0]
    # insertion_ours_images.append(insertion_image)
    # deletion_ours_images.append(image - insertion_image)
    # for smdl_sub_mask in smdl_mask[1:]:
    #     insertion_image = insertion_image.copy() + smdl_sub_mask
    #     insertion_ours_images.append(insertion_image)
    #     deletion_ours_images.append(image - insertion_image)
    # insertion_ours_images.append(image)
    # deletion_ours_images.append(image - image)
    
    insertion_explanation_images_input = load_image(
        np.array(insertion_explanation_images)[..., ::-1]
    )
    deletion_explanation_images_input = load_image(
        np.array(deletion_explanation_images)[..., ::-1]
    )

    insertion_ours_images_input = load_image(
        np.array(insertion_ours_images)[..., ::-1]
    )
    deletion_ours_images_input = load_image(
        np.array(deletion_ours_images)[..., ::-1]
    )
    
    insertion_explanation_images_input_results = softmax(model(insertion_explanation_images_input)/0.05)[:,class_index]
    
    insertion_ours_images_input_results = softmax(model(insertion_ours_images_input)/0.05)[:,class_index] 
    
    explanation_best_index = np.argmax(insertion_explanation_images_input_results)
    ours_best_index = np.argmax(insertion_ours_images_input_results)
    
    frames = []
    x = list(np.linspace(0,1,steps+1))
    for i in range(len(x)):
        fig, [ax1, ax2, ax3] = plt.subplots(1,3, gridspec_kw = {'width_ratios':[1, 1, 1.5]}, figsize=(30,8))
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.imshow(insertion_explanation_images[i][...,::-1])
        ax1.set_title(method_name, fontsize=54)

        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.imshow(insertion_ours_images[i][...,::-1])
        ax2.set_title('+ Ours', fontsize=54)

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
        plt.title('Insertion', fontsize=54)
        plt.ylabel('Recognition Score', fontsize=44)
        plt.xlabel('Percentage of image revealed', fontsize=44)

        x_ = x[:i+1]
        explanation_y = insertion_explanation_images_input_results.numpy()[:i+1]
        plt.plot(x_, explanation_y, color='orange', linewidth=3.5)  # 绘制曲线

        ours_y = insertion_ours_images_input_results.numpy()[:i+1]
        plt.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # 绘制曲线

        plt.legend(['{}'.format(method_name), "+ Ours"], fontsize=40, loc="upper right")
        plt.scatter(x_[-1], explanation_y[-1], color='orange', s=54)  # 绘制最新点
        plt.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # 绘制最新点

        if i == len(x)-1:
            kernel = np.ones((3, 3), dtype=np.uint8)
            plt.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)  # 绘制红色曲线
            
            # Ours
            mask = (image - insertion_ours_images[ours_best_index]).mean(-1)
            mask[mask>0] = 1

            dilate = cv2.dilate(mask, kernel, 3)
            # erosion = cv2.erode(dilate, kernel, iterations=3)
            # dilate = cv2.dilate(erosion, kernel, 2)
            edge = dilate - mask
            # erosion = cv2.erode(dilate, kernel, iterations=1)

            image_debug = image.copy()

            image_debug[mask>0] = image_debug[mask>0] * 0.5
            image_debug[edge>0] = np.array([0,0,255])
            ax2.imshow(image_debug[...,::-1])

            # Attribution
            mask = (image - insertion_explanation_images[ours_best_index]).mean(-1)
            mask[mask>0] = 1
            dilate = cv2.dilate(mask, kernel, 3)
            edge = dilate - mask
            image_debug_exp = image.copy()
            image_debug_exp[mask>0] = image_debug_exp[mask>0] * 0.5
            image_debug_exp[edge>0] = np.array([0,0,255])
            ax1.imshow(image_debug_exp[...,::-1])

        plt.savefig("gif_tmp.png", bbox_inches='tight')
        img_frame = cv2.imread("gif_tmp.png")
        frames.append(img_frame.copy()[...,::-1])
    print("Highest confidence:{}, final confidence:{}".format(insertion_ours_images_input_results.numpy().max(), insertion_ours_images_input_results[-1]))
    for j in range(20):
        frames.append(img_frame.copy()[...,::-1])
    
    imageio.mimsave(image_path.split("/")[-1].replace(".jpg", ".gif"), frames, 'GIF', duration=0.1)  
    

main()