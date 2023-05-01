import scipy

import os
import numpy as np
import cv2

import tensorflow as tf
import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')
from matplotlib.colors import ListedColormap

import tensorflow_probability as tfp


def load_image(path, size=224):
    img = cv2.resize(cv2.imread(path)[...,::-1], (size, size))
    return img.astype(np.float32)

def set_size(w,h):
    """Set matplot figure size"""
    plt.rcParams["figure.figsize"] = [w,h]

def show(img, p=False, smooth=False, minn=None, maxx = None, **kwargs):
    """ Display numpy/tf tensor """ 
    img = np.array(img, dtype=np.float32)

    # check if channel first
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)

    # check if cmap
    if img.shape[-1] == 1:
        img = img[:,:,0]

    # normalize
    if minn is None:
        if img.max() > 1 or img.min() < 0:
            img -= img.min(); img/=img.max()

    # check if clip percentile
    if p is not False:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))
    
    if smooth and len(img.shape) == 2:
        img = gaussian_filter(img, smooth)

    if minn is not None:
        plt.imshow(img, vmin = minn, vmax =maxx, **kwargs)
    else:
        plt.imshow(img,  **kwargs)
    plt.axis('off')
    plt.grid(None)
    
def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = ListedColormap(alpha_cmap)

    return alpha_cmap

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def norm(image):
    """
    :param image: [H,W]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    
    return image

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap