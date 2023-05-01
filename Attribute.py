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

grid_size = 7
nb_forward = 400

attribute_threshold = 0.5

mt = "VGGFace2"

if mt == "VGGFace2":
    # VGGFace2
    ID_path = [
        'motivation/images/VGGFace2/ID/n000307',
        'motivation/images/VGGFace2/ID/n000309',
        'motivation/images/VGGFace2/ID/n000337',
        'motivation/images/VGGFace2/ID/n000353',
        'motivation/images/VGGFace2/ID/n000359',
        'motivation/images/VGGFace2/ID/n003021',
        'motivation/images/VGGFace2/ID/n003197',
        'motivation/images/VGGFace2/ID/n005546',
        'motivation/images/VGGFace2/ID/n006579',
        'motivation/images/VGGFace2/ID/n006634',
    ]

    model_path = "ckpt/AttributeNet-CelebA.onnx"
    from models_onnx.Attr_CelebA import AttributeModel
    save_dir = "motivation/results/VGGFace2/Attribute"
    mkdir(save_dir)

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

elif mt == "CelebA":
    # Celeb-A
    ID_path = [
        'motivation/images/Celeb-A/ID/32',
        'motivation/images/Celeb-A/ID/45',
        'motivation/images/Celeb-A/ID/137', 
        'motivation/images/Celeb-A/ID/207',
        'motivation/images/Celeb-A/ID/241', 
        'motivation/images/Celeb-A/ID/325', 
        'motivation/images/Celeb-A/ID/423', 
        'motivation/images/Celeb-A/ID/535', 
        'motivation/images/Celeb-A/ID/620', 
        'motivation/images/Celeb-A/ID/768', 
        'motivation/images/Celeb-A/ID/824', 
        'motivation/images/Celeb-A/ID/922', 
    ]
    model_path = "ckpt/AttributeNet-VGGFace2.onnx"
    from models_onnx.Attr_VGG import AttributeModel
    save_dir = "motivation/results/Celeb-A/Attribute"
    mkdir(save_dir)

    attribute_set = ["Male", "Female", 
                     "Young", "Middle Aged", "Senior", 
                     "Asian", "White", "Black", 
                     "Brown Eyes","Bags Under Eyes",
                     "Bushy Eyebrows","Arched Eyebrows", 
                     "Big Lips",
                     "Big Nose","Pointy Nose"]

    def prepare_image(path):
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        # image = image.transpose(2,0,1)
        # image = np.array([image])
        return image

def main():
    tf_model = AttributeModel(model_path)
    tf_model.set_idx_list(attribute_set)

    hsic_explainer = FaceHsicAttributionMethod(tf_model, 
                                      grid_size = grid_size, 
                                      nb_design = nb_forward , 
                                      sampler = HsicLHSSampler(binary=True), 
                                      estimator = HsicEstimator(kernel_type="binary"),
                                      perturbation_function = 'inpainting',
                                      batch_size = 32)

    for id_person in ID_path:
        # Image list
        image_names = os.listdir(id_person)

        # Each ID person's save dir
        id_save_dir = os.path.join(save_dir, id_person.split("/")[-1])
        mkdir(id_save_dir)
        # Visualization save dir
        visualization_save_dir = os.path.join(id_save_dir, "visualization")
        mkdir(visualization_save_dir)
        # Explanation save dir
        explanation_save_dir = os.path.join(id_save_dir, "explanation")
        mkdir(explanation_save_dir)

        # json file
        Json_file = {}

        for image_name in image_names:
            sub_json = {}
            # image path
            image_path = os.path.join(id_person, image_name)
            
            # prepare image
            X_raw = np.array([prepare_image(image_path)])
            
            # The predicted attribute
            predict = tf_model(X_raw, True)[0]
            predict = (predict > attribute_threshold).astype(int)

            labels = np.argwhere(predict==1).flatten()
            labels_ohe = tf.one_hot(labels, len(attribute_set))
            
            attr = np.array(tf_model.facial_attribute)[labels]
            
            sub_json["attribute"] = attr.tolist()
            
            # HSIC Attribution Map
            Input = np.array([prepare_image(image_path) for i in attr])
            explanations = hsic_explainer(Input, labels_ohe)
            explanations = np.array(explanations)

            np.save(os.path.join(explanation_save_dir, image_name.replace(".jpg", "")), explanations)

            Json_file[image_name] = sub_json

            for idx, attr_ in enumerate(attr):
                attr_image_save_dir = os.path.join(visualization_save_dir, attr_)
                mkdir(attr_image_save_dir)

                mask = explanations[idx]
                # norm
                mask = norm(mask)
                image = cv2.imread(image_path)
                image = cv2.resize(image, mask.shape)
                save_image_name = image_name

                cam, heatmap = gen_cam(image, mask)
                cv2.imwrite(os.path.join(attr_image_save_dir, save_image_name), cam)
        with open(os.path.join(id_save_dir, "Record.json"), "w") as f:
            f.write(json.dumps(Json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
    return

if __name__ == '__main__':

    main()


