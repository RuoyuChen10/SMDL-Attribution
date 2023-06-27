import cv2
import numpy as np
import torch.nn.functional as F

from models.face_parser import FaceParser, remove, read_img
from utils import *

# mt = "VGGFace2"
mt = "CelebA"

if mt == "VGGFace2":
    image_dir_path = "motivation/images/VGGFace2/ID"
    eval_path = "motivation/results/VGGFace2/ID"
elif mt == "CelebA":
    image_dir_path = "motivation/images/Celeb-A/ID"
    eval_path = "motivation/results/Celeb-A/ID"

seg_part = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']
seg_select_part = ['mouth', 'eyebrows', 'eyes', 'nose']
mask_id = [1, 2, 3, 5]

threshold = 0.4

def test():
    model = FaceParser()
    model.eval()
    image_path = "image/n000003-0002_01.jpg"

    # 'background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface'
    image = read_img(image_path)
    parsed_face = model(image)
    mask = (parsed_face.cpu().numpy()[0] > threshold).astype(int)# * 255

    org_image = cv2.imread(image_path)
    org_image = cv2.resize(org_image, (512,512))
    for i in mask_id:
        org_image = org_image * (1 - mask[i])[:, :, np.newaxis]
        cv2.imwrite(seg_part[i]+".jpg", mask[i]*255)
    cv2.imwrite("result.jpg", org_image)

def norm(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    return image

def face_part_activation(explanation, masks):
    """
    Compute the consistency
    : explanation: the interpretation mask  (112, 112)
    : masks: the part mask  (part_number, 112, 112)
    """
    activation_part = masks * explanation # (part_number, 112, 112)
    
    part_region_area = masks.sum(-1).sum(-1)
    if 0 in part_region_area:
        return None, False
    
    activation_total_score = activation_part.sum(-1).sum(-1)
    
    activation_vector = activation_total_score / part_region_area
    return activation_vector, True

def consistency(activation_vectors):
    """
    activation_vectors: List(
        vector1
        vector2
        ...
    )
    """
    scores = []
    for i in range(len(activation_vectors) - 1):
        for j in range(i+1, len(activation_vectors)):
            x_norm = activation_vectors[i] / np.linalg.norm(activation_vectors[i], ord=2, axis=0)
            y_norm = activation_vectors[j] / np.linalg.norm(activation_vectors[j], ord=2, axis=0)
            scores.append(np.dot(x_norm, y_norm))
    
    average_score = np.array(scores).mean()

    return average_score

def most_salient_region(activation_vectors):
    """
    activation_vectors: List(
        vector1
        vector2
        ...
    )
    """
    activation_vectors = np.array(activation_vectors)
    activation_vectors = activation_vectors.mean(0)
    # judgement
    assert activation_vectors.shape[0] == len(seg_select_part)
    region_id = np.argmax(activation_vectors)
    region = seg_select_part[region_id]
    return region

def main():
    device = "cuda:0"
    # Init model
    model = FaceParser()
    model.eval()
    model.to(device)

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
    peoples = os.listdir(eval_path)
    for people in peoples:
        # each people results dir
        people_path = os.path.join(eval_path, people)
        image_txt = os.path.join(people_path, "image.txt")
        explanation_masks = np.load(
            os.path.join(people_path, "explanation.npy")
        )
        # original image dir
        image_dir = os.path.join(image_dir_path, people)

        with open(image_txt, "r") as f:
            image_names = f.read().split('\n')
            while "" in image_names:
                image_names.remove("")
        
        activation_vectors = []
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(image_dir, image_name)
            
            # Face paser
            image = read_img(image_path)
            parsed_face = model(image.to(device))
            # mask = (parsed_face.cpu().numpy()[0] > threshold).astype(int)   # * 255
            
            masks = F.interpolate(parsed_face, explanation_masks.shape[1:3], mode="bilinear")
            masks = (masks.cpu().numpy()[0] > threshold).astype(int)
            masks = np.array([masks[idx] for idx in mask_id])
            
            explanation = norm(explanation_masks[i])
            
            activation_vector, get_activation_vector = face_part_activation(explanation, masks)
            if get_activation_vector:
                activation_vectors.append(activation_vector)
        
        consistency_score = consistency(activation_vectors)
        region = most_salient_region(activation_vectors)

        print("Person {}'s consistency score: {}, most salient region: {}.".format(people, consistency_score, region))
        #     org_image = cv2.imread(image_path)
        #     org_image = cv2.resize(org_image, (112,112))
        #     for j, mask in enumerate(masks):
        #         org_image = org_image * (1-mask)[:, :, np.newaxis]
        #         cv2.imwrite(seg_select_part[j]+".jpg", mask*255)
        #     cv2.imwrite("result.jpg", org_image)
        #     break
        # break

    return

if __name__ == '__main__':
    main()