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