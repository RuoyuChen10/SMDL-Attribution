# qmc module, ensure scipy correct version before anything

import scipy

import os
import numpy as np
import cv2

import tensorflow as tf
import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import tensorflow_probability as tfp

import xplique
from xplique.attributions import *
from xplique.metrics import *

from xplique_addons import *
from utils import *

red_tr    = get_alpha_cmap('Reds')

from models.submodular import SubModular

# batch_size = 32

# images_classes = [
#                   ('assets/fox.png', 278),
#                   ('assets/leopard.png', 288),
#                   ('assets/polar_bear.png', 296),
#                   ('assets/snow_fox.png', 279),
# ]

# X_raw = np.array([load_image(p) for p, y in images_classes])
# Y_true = np.array([y for p, y in images_classes])


# model = tf.keras.applications.ResNet50V2()
# model.layers[-1].activation = tf.keras.activations.linear
# inputs =  tf.keras.applications.resnet_v2.preprocess_input(np.array([x.copy() for x in X_raw], copy=True))  # 4*224*224*3

# labels = np.argmax(model.predict(inputs, batch_size=batch_size), axis=-1)
# labels_ohe = tf.one_hot(labels, 1000)

# grid_size = 7
# nb_forward = 1536


# hsic_explainer = HsicAttributionMethod(model, 
#                                       grid_size = grid_size, 
#                                       nb_design = nb_forward , 
#                                       sampler = HsicLHSSampler(binary=True), 
#                                       estimator = HsicEstimator(kernel_type="binary"),
#                                       perturbation_function = 'inpainting',
#                                       batch_size = 256)

# explanations = hsic_explainer(inputs, labels_ohe)
# explanations = np.array(explanations)   # shape (4, 224, 224)

x = SubModular()