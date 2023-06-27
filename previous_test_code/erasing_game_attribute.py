import scipy

import os
import numpy as np
import cv2

import tensorflow as tf
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import tensorflow_probability as tfp

import xplique
from xplique.attributions import *
from xplique.metrics import *

from xplique_addons import *
from utils import *

from explanation.face_interpret import FaceHsicAttributionMethod
from models_onnx.ID_NET import ONNX_Face_Recognition

import onnx_tf.backend
import onnx

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF
import torch.nn.functional as F

import json

erasing_threshold = 0.3
attribute_threshold = 0.5

mt = "VGGFace2-test"

if mt == "VGGFace2":
    Attribute_results_path = "motivation/results/VGGFace2/Attribute"
    Attribute_image_path = "motivation/images/VGGFace2/ID/"
    ID_names = [
        'n000307',
        'n000309',
        'n000337',
        'n000353',
        'n000359', 
        'n003021',
        'n003197',
        'n005546',
        'n006579', 
        'n006634',
    ]
    model_path = "ckpt/AttributeNet-CelebA.onnx"

    from models_onnx.Attr_CelebA import AttributeModel
    attribute_set = [
        'male', 'female', 
        'young', 'old',
        'arched_eyebrows', 'bushy_eyebrows',
        'mouth_slightly_open', 'big_lips',
        'big_nose', 'pointy_nose',
        'bags_under_eyes', 'narrow_eyes'
    ]

elif mt == "VGGFace2-test":
    Attribute_results_path = "motivation/results/VGGFace2-test/Attribute"
    Attribute_image_path = "motivation/images/VGGFace2/Attribute/VGGFace2-test"
    ID_names = os.listdir("motivation/results/VGGFace2-test/Attribute")

    model_path = "ckpt/AttributeNet-CelebA.onnx"

    from models_onnx.Attr_CelebA import AttributeModel
    attribute_set = [
        'male', 'female', 
        'young', 'old',
        'arched_eyebrows', 'bushy_eyebrows',
        'mouth_slightly_open', 'big_lips',
        'big_nose', 'pointy_nose',
        'bags_under_eyes', 'narrow_eyes'
    ]

def interpolate(img, size):
    if type(size) == tuple:
        assert size[0] == size[1]
        size = size[0]
    
    orig_size = img.size(3)
    if size < orig_size:
        mode = 'area'
    else:
        mode = 'bilinear'
    return F.interpolate(img, (size, size), mode=mode)

def prepare_image(path):
    img = Image.open(path).convert('RGB')
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 224:
        img = interpolate(img, 224)
    img = img.permute(0, 2, 3, 1)
    return img[0].numpy()

def convert_prepare_image(image, size=224):
    img = Image.fromarray(image)
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 224:
        img = interpolate(img, 224)
    img = img.permute(0, 2, 3, 1)

    return img[0].numpy()

def norm(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    return image

def test():
    masks = np.load("motivation/results/VGGFace2/ID/n000307/explanation.npy")
    image_path = "motivation/images/VGGFace2/ID/n000307/0283_01.jpg"
    
    mask = norm(masks[1])[:, :, np.newaxis]
    
    image = cv2.imread(image_path)

    masked_image = image * (mask < erasing_threshold).astype(int)

    cv2.imwrite("./result-{}.jpg".format(erasing_threshold), masked_image)

    return 

def main():
    """
    - Eval Path
        - people1
            - CAM
                - image 1
                - image 2
                - ...
            - explanation.npy
            - image.txt
        - people2
        - ...
    """
    tf_model = AttributeModel(model_path)
    tf_model.set_idx_list(attribute_set)

    acc = np.zeros(len(attribute_set))
    number = np.zeros(len(attribute_set))

    for id_person in ID_names:
        id_person_path = os.path.join(Attribute_results_path, id_person)

        # Read the json file
        with open(os.path.join(id_person_path, 'Record.json'),'r',encoding = 'utf-8') as fp:
            json_data = json.load(fp)
        
        explanation_masks_save_dir = os.path.join(id_person_path, "explanation")
        explanation_masks_names = os.listdir(explanation_masks_save_dir)

        for explanation_masks_name in explanation_masks_names:
            attributes = json_data[explanation_masks_name.replace(".npy", ".jpg")]['attribute']
            # e.g., attributes = ['female', 'young', 'mouth_slightly_open']
            
            explanation_masks = np.load(
                os.path.join(explanation_masks_save_dir, explanation_masks_name))
            
            image_path = os.path.join(
                os.path.join(Attribute_image_path, id_person),
                explanation_masks_name.replace(".npy", ".jpg")
            )
            for i, attribute in enumerate(attributes):
                # Read Image
                image = prepare_image(image_path)

                attribute_idex = attribute_set.index(attribute)
                number[attribute_idex] += 1

                # load special attribute's HSIC mask
                mask_attribute = norm(explanation_masks[i])[:, :, np.newaxis]
                masked_image = image * (mask_attribute < erasing_threshold).astype(int)

                input_image = np.array([masked_image.astype(np.float32)])

                predicted_score = tf_model(input_image)[0][attribute_idex]
                
                if predicted_score > attribute_threshold:
                    acc[attribute_idex] += 1

    print("Erasing Rate {}, ACC {}, number {}".format(erasing_threshold, acc/number, number))

    return 

if __name__ == '__main__':

    main()