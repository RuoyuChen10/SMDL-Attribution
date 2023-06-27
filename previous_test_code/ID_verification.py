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
from models_onnx.ID_VER import ONNX_Face_Verification

import onnx_tf.backend
import onnx

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

red_tr    = get_alpha_cmap('Reds')

mt = "VGGFace2"

if mt == "VGGFace2":
    ID_path = [
        ("motivation/images/VGGFace2/ID/n000307/0002_01.jpg", "motivation/images/VGGFace2/ID/n003021/0016_02.jpg"),
    ]
    
    model_path = "ckpt/ArcFace-VGGFace2-R50-8631-verification.onnx"
    
transforms = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def prepare_image(path, size=112):
    img = Image.open(path)
    img = transforms(img).numpy()
    img = img.transpose(1, 2, 0)
    
    return img

def same_id_test():
    ID_path = [
        ('motivation/images/VGGFace2/ID/n000307', "0005_01.jpg"),
        ('motivation/images/VGGFace2/ID/n000309', "0025_01.jpg"),
        ('motivation/images/VGGFace2/ID/n000337', "0029_01.jpg"),
        ('motivation/images/VGGFace2/ID/n000353', "0025_01.jpg"),
        ('motivation/images/VGGFace2/ID/n000359', "0015_01.jpg"),
        ('motivation/images/VGGFace2/ID/n003021', "0016_02.jpg"),
        ('motivation/images/VGGFace2/ID/n003197', "0155_02.jpg"),
        ('motivation/images/VGGFace2/ID/n005546', "0032_01.jpg"),
        ('motivation/images/VGGFace2/ID/n006579', "0042_01.jpg"),
        ('motivation/images/VGGFace2/ID/n006634', "0040_01.jpg"),
    ]
    
    save_dir = "results-ver/VGGFace2/"
    mkdir(save_dir)
    
    grid_size = 7
    nb_forward = 200
    
    tf_model = ONNX_Face_Verification(model_path)
    tf_model.verification_direction = True
    
    hsic_explainer = FaceHsicAttributionMethod(tf_model, 
                                      grid_size = grid_size, 
                                      nb_design = nb_forward , 
                                      sampler = HsicLHSSampler(binary=True), 
                                      estimator = HsicEstimator(kernel_type="binary"),
                                      perturbation_function = 'inpainting',
                                      batch_size = 32)

    for id_person, support_image_path in ID_path:
        # Image list
        images = os.listdir(id_person)
        images.remove(support_image_path)
        query_images = np.array([prepare_image(os.path.join(id_person, p)) for p in images])
        
        support_image = np.array([prepare_image(os.path.join(id_person, support_image_path))])
        
        # filling support feature
        hsic_explainer.model.fill_verified_face(support_image)
        
        labels = np.array([0 for p in images])
        labels_ohe = tf.one_hot(labels, 1)
        
        explanations = hsic_explainer(query_images, labels_ohe)
        explanation = np.array(explanations)   # (batch, weight, height)
        
        image_save_dir = os.path.join(save_dir, id_person.split("/")[-1])
        mkdir(image_save_dir)
        
        for idx in range(explanations.shape[0]):
            mask = explanation[idx]
            # norm
            mask = norm(mask)

            image = cv2.imread(
                os.path.join(id_person, images[idx])
            )
            
            cam, heatmap = gen_cam(image, mask)
            cv2.imwrite(os.path.join(image_save_dir, images[idx]), cam)

def main():
    grid_size = 7
    nb_forward = 200
    
    tf_model = ONNX_Face_Verification(model_path)
    
    hsic_explainer = FaceHsicAttributionMethod(tf_model, 
                                      grid_size = grid_size, 
                                      nb_design = nb_forward , 
                                      sampler = HsicLHSSampler(binary=True), 
                                      estimator = HsicEstimator(kernel_type="binary"),
                                      perturbation_function = 'inpainting',
                                      batch_size = 32)
    
    for query_image_path, support_image_path in ID_path:
        # image
        query_image = np.array([prepare_image(query_image_path)])
        
        support_image = np.array([prepare_image(support_image_path)])
        
        # filling support feature
        hsic_explainer.model.fill_verified_face(support_image)
        
        labels = [0]
        labels_ohe = tf.one_hot(labels, 1)
        
        explanations = hsic_explainer(query_image, labels_ohe)
        explanation = np.array(explanations)   # (batch, weight, height)
        
        for idx in range(explanation.shape[0]):
            mask = explanation[idx]
            
            # norm
            mask = norm(mask)
            
            image = cv2.imread(
                query_image_path
            )
            cam, heatmap = gen_cam(image, mask)
            cv2.imwrite("result.jpg", cam)



if __name__ == '__main__':

    # main()
    
    same_id_test()
    