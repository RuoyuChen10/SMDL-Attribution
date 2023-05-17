import scipy

import os
import numpy as np
import cv2

from matplotlib import pyplot as plt
plt.style.use('seaborn')

from xplique.attributions import *
from xplique.metrics import *

from xplique_addons import *
from utils import *

from models_onnx.ID_NET import ONNX_Face_Recognition

import numpy as np
from PIL import Image
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
    model_path = "ckpt/ArcFace-VGGFace2-R50-8631.onnx"
    class_num = 8631

    transforms = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

erasing_threshold = 0.3

def prepare_image(path, size=112):
    img = Image.open(path)
    img = transforms(img).numpy()
    img = img.transpose(1, 2, 0)
    
    return img

def convert_prepare_image(image, size=112):
    image = cv2.resize(image, (size, size))
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = transforms(img).numpy()
    img = img.transpose(1, 2, 0)
    return img

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
    tf_model = ONNX_Face_Recognition(model_path)

    acc = 0
    number = 0

    for id_person, gt_label in ID_names:
        id_person_path = os.path.join(ID_results_path, id_person)

        erase_dir = os.path.join(id_person_path, "Erase_image")
        mkdir(erase_dir)
        erase_id_image_dir = os.path.join(erase_dir, "Erasing_Rate_{}".format(erasing_threshold))
        mkdir(erase_id_image_dir)
        
        if(os.path.isfile(os.path.join(erase_dir, "error-{}.txt".format(erasing_threshold)))):
            os.remove(os.path.join(erase_dir, "error-{}.txt".format(erasing_threshold)))

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
        
        acc_id = 0
        number_id = 0 

        # Loop image names
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(os.path.join(ID_image_path, id_person), image_name)
            image = cv2.imread(image_path)

            mask = norm(explanation_masks[i])[:, :, np.newaxis]
            masked_image = image * (mask < erasing_threshold).astype(int)
            
            input_image = np.array([convert_prepare_image(masked_image.astype(np.uint8))])

            predicted_labels = np.argmax(tf_model(input_image), axis=-1)
            
            number += 1
            number_id += 1
            if predicted_labels[0] == gt_label:
                acc += 1
                acc_id += 1
            else:
                with open(os.path.join(erase_dir, "error-{}.txt".format(erasing_threshold)), 'a') as file:
                    file.write(os.path.join(id_person, image_name) + '\n')
            cv2.imwrite(os.path.join(erase_id_image_dir, image_name), 
                        masked_image.astype(np.uint8))
        print("Erasing Rate {}, People {}, ACC {}".format(erasing_threshold, id_person, acc_id/number_id))

    print("Erasing Rate {}, ACC {}, number {}".format(erasing_threshold, acc/number, number))

    return 

if __name__ == '__main__':

    main()