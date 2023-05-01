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

red_tr    = get_alpha_cmap('Reds')

# mt = "VGGFace2"
mt = "CelebA"

if mt == "VGGFace2":
    # VGGFace2
    ID_path = [
                    ('motivation/images/VGGFace2/ID/n000307', 290),
                    ('motivation/images/VGGFace2/ID/n000309', 292),
                    ('motivation/images/VGGFace2/ID/n000337', 320),
                    ('motivation/images/VGGFace2/ID/n000353', 336),
                    ('motivation/images/VGGFace2/ID/n000359', 342),
                    ('motivation/images/VGGFace2/ID/n003021', 2816),
                    ('motivation/images/VGGFace2/ID/n003197', 2984),
                    ('motivation/images/VGGFace2/ID/n005546', 5144),
                    ('motivation/images/VGGFace2/ID/n006579', 6103),
                    ('motivation/images/VGGFace2/ID/n006634', 6156),
    ]

    model_path = "ckpt/ArcFace-VGGFace2-R50-8631.onnx"
    save_dir = "motivation/results/VGGFace2"
    class_num = 8631
elif mt == "CelebA":
    # Celeb-A
    ID_path = [
                    ('motivation/images/Celeb-A/ID/32', 32),
                    ('motivation/images/Celeb-A/ID/45', 45),
                    ('motivation/images/Celeb-A/ID/137', 137),
                    ('motivation/images/Celeb-A/ID/207', 207),
                    ('motivation/images/Celeb-A/ID/241', 241),
                    ('motivation/images/Celeb-A/ID/325', 325),
                    ('motivation/images/Celeb-A/ID/423', 423),
                    ('motivation/images/Celeb-A/ID/535', 535),
                    ('motivation/images/Celeb-A/ID/620', 620),
                    ('motivation/images/Celeb-A/ID/768', 768),
                    ('motivation/images/Celeb-A/ID/824', 824),
                    ('motivation/images/Celeb-A/ID/922', 922),
    ]
    model_path = "ckpt/ArcFace-CelebA-R50-10177.onnx"
    save_dir = "motivation/results/Celeb-A"
    class_num = 10177

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

def main():
    grid_size = 7
    nb_forward = 200

    tf_model = ONNX_Face_Recognition(model_path)

    hsic_explainer = FaceHsicAttributionMethod(tf_model, 
                                      grid_size = grid_size, 
                                      nb_design = nb_forward , 
                                      sampler = HsicLHSSampler(binary=True), 
                                      estimator = HsicEstimator(kernel_type="binary"),
                                      perturbation_function = 'inpainting',
                                      batch_size = 32)
    # Every People
    for id_person, gt_label in ID_path:
        # Image list
        images = os.listdir(id_person)

        X_raw = np.array([prepare_image(os.path.join(id_person, p)) for p in images])
        labels = np.argmax(tf_model(X_raw), axis=-1)
        labels_ohe = tf.one_hot(labels, class_num)
        
        explanations = hsic_explainer(X_raw, labels_ohe)
        explanation = np.array(explanations)   # (batch, weight, height)

        # path to save the same people
        id_save_dir = os.path.join(save_dir, id_person.split("/")[-1])
        mkdir(id_save_dir)
        id_image_save_dir = os.path.join(id_save_dir, "CAM")
        mkdir(id_image_save_dir)

        with open(os.path.join(id_save_dir, "image.txt"), 'a') as file:
            for img_ in images:
                file.write(img_+'\n')

        np.save(os.path.join(id_save_dir, "explanation"), explanation)

        for idx in range(explanations.shape[0]):
            mask = explanation[idx]
            # norm
            mask = norm(mask)

            image = cv2.imread(
                os.path.join(id_person, images[idx])
            )
            save_image_name = images[idx].replace(".jpg", "-predict_id_{}-ground_truth_{}.jpg".format(labels[idx], gt_label))

            cam, heatmap = gen_cam(image, mask)

            cv2.imwrite(os.path.join(id_image_save_dir, save_image_name), cam)

if __name__ == '__main__':

    main()