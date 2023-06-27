# qmc module, ensure scipy correct version before anything

import scipy

import os
import numpy as np
import cv2
    
# import tensorflow as tf
import subprocess
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')
# import tensorflow_probability as tfp

# import xplique
# from xplique.attributions import *
# from xplique.metrics import *

# from xplique_addons import *
from utils import *

red_tr    = get_alpha_cmap('Reds')

from models.submodular import FaceSubModularExplanation

import torchvision.transforms as transforms

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
    interval = 0.01

smdl = FaceSubModularExplanation()

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
        
        for erasing_threshold in (np.arange(0, 1, interval) + interval):
            masked_image = image * (mask < erasing_threshold).astype(int) * (mask > erasing_threshold-interval).astype(int)
            mask_images.append(masked_image.astype(np.uint8))
            
            components_image_list.append(masked_image.astype(np.uint8))

        smdl(components_image_list)
        
        break
    break
