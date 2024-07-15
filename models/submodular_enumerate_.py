# -*- coding: utf-8 -*-  

"""
Created on 2023/5/9

@author: Ruoyu Chen
"""

import json
import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from itertools import combinations
from collections import OrderedDict

from .iresnet import iresnet50
from .iresnet_edl import iresnet100
from .evidential import relu_evidence, exp_evidence

class SubModular():
    def __init__(self, cfg_path="models/submodular_cfg.json", combination_number_k = 5):
        super(SubModular, self).__init__()
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
            
        self.combination_number_k = combination_number_k

        self.device = torch.device(self.cfg["device"])

        self.uncertainty_model = self.define_uncertrainty_network(
            self.cfg["uncertainty_model"]["num_classes"], self.cfg["uncertainty_model"]["model_path"])
    
        self.face_recognition_model = self.define_recognition_model(
            self.cfg["face_recognition_model"]["num_classes"], self.cfg["face_recognition_model"]["model_path"])
        
        self.lambda1 = 1
        self.lambda2 = 0.5
        self.lambda3 = 0.5

    def define_recognition_model(self, num_classes, pretrained_path):
        """
        init the face recognition model
        """
        model = iresnet50(num_classes)

        if pretrained_path is not None and os.path.exists(pretrained_path):
            model_dict = model.state_dict()
            pretrained_param = torch.load(pretrained_path)

            try:
                pretrained_param = pretrained_param.state_dict()
            except:
                pass
                
            new_state_dict = OrderedDict()
            for k, v in pretrained_param.items():
                if k in model_dict:
                    new_state_dict[k] = v
                    # print("Load parameter {}".format(k))
                elif k[7:] in model_dict:
                    new_state_dict[k[7:]] = v
                    # print("Load parameter {}".format(k[7:]))
                else:
                    print("Parameter {} has not been load".format(k))
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            print("Success load pre-trained face recognition model {}".format(pretrained_path))
        else:
            print("not load pretrained")
        
        model.eval()
        model.to(self.device)

        return model

    def define_uncertrainty_network(self, num_classes, pretrained_path):
        """
        Init the uncertainty model
        """
        model = iresnet100(num_classes)

        if pretrained_path is not None and os.path.exists(pretrained_path):
            model_dict = model.state_dict()
            pretrained_param = torch.load(pretrained_path)

            try:
                pretrained_param = pretrained_param.state_dict()
            except:
                pass
                
            new_state_dict = OrderedDict()
            for k, v in pretrained_param.items():
                if k in model_dict:
                    new_state_dict[k] = v
                    # print("Load parameter {}".format(k))
                elif k[7:] in model_dict:
                    new_state_dict[k[7:]] = v
                    # print("Load parameter {}".format(k[7:]))
                else:
                    print("Parameter {} has not been load".format(k))
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            print("Success load pre-trained uncertainty model {}".format(pretrained_path))
        else:
            print("not load pretrained")
        
        model.eval()
        model.to(self.device)

        return model

    def compute_uncertainty(self, input_face_images, scale = 5):
        """
        Compute the uncertainty of the model
        input: torch.Size(batch, 3, w, h)
        """
        with torch.no_grad():
            output = self.uncertainty_model(input_face_images)
        evidence = exp_evidence(scale * output)
        alpha = evidence + 1
        uncertainty = self.cfg["uncertainty_model"]["num_classes"] / torch.sum(alpha, dim=1, keepdim=True)

        return uncertainty.reshape(-1)
    
    def compute_redundancy_score(self, face_features):
        """
        Computes Redundancy Score: The point should be distant from all the other elements in the subset.
        face_features: torch.Size(batch, d)
        """
        if self.cfg["redundancy_distance_metric"] == "cosine":
            norm_feature = F.normalize(face_features, p=2, dim=1)
            # Consine Similarity
            consine_similarity = torch.mm(norm_feature, norm_feature.t())
            consine_similarity = torch.clamp(consine_similarity, min=-1, max=1)
            # Normlize 0-1
            consine_dist = torch.arccos(consine_similarity) / math.pi

            r_scores = torch.min(consine_dist + torch.eye(norm_feature.shape[0]).to(self.device), -1)[0]
        return r_scores.mean() # tensor([0.5032, 0.4973, 0.4973])
    
    def proccess_compute_repudancy_score(self, components_image_feature, combination_list):
        """
        Compute each S's r score
        """
        r_scores = []
        for sub_index in combination_list:
            sub_feature_set = components_image_feature[np.array(sub_index), :]
            r_score = self.compute_redundancy_score(sub_feature_set)
            r_scores.append(r_score.cpu().numpy())

        return np.array(r_scores)

    def compute_mean_closeness_score(self, face_features, source_face_feature=None):
        """
        Computes Mean Closeness score: The new datapoint should be close to the class mean
        face_features: torch.Size(batch, d)
        """
        if self.cfg["redundancy_distance_metric"] == "cosine":
            norm_feature = F.normalize(face_features, p=2, dim=1)
            if source_face_feature == None:
                mean_feature = F.normalize(face_features.mean(0, keepdim=True), p=2, dim=1)
            else:
                mean_feature = F.normalize(source_face_feature, p=2, dim=1)

            consine_similarity = torch.mm(norm_feature, mean_feature.t())
            consine_dist = torch.arccos(consine_similarity) / math.pi

            mc_score = consine_dist.reshape(-1)
        return mc_score
    
    def normalize(self, A):
        """
        A: torch.Size(D)
        """
        return A / A.sum()
    
    def combination_mask(self, image_list, k = None):
        """permutation
        :image_list: [image1, image2, ...]
        """
        org_img_num = len(image_list)
        index = list(range(org_img_num))
        
        if k == None:
            k = org_img_num + 1
        else:
            k = k + 1
        
        combination_image_list = []
        combination_list = []
        
        # number for combination
        for comb_num in range(2, k):
            # Combination
            sub_index_list = list(combinations(index, comb_num))
            for sub_index in sub_index_list:
                # combinate the masked images
                combination_mask_image = np.zeros_like(image_list[0])
                for idx in sub_index:
                    combination_mask_image = combination_mask_image + image_list[idx]
                combination_image_list.append(combination_mask_image)
                combination_list.append(sub_index)

        return combination_image_list, combination_list
    
    def _get_combination_list(self, image_list, k = None):
        """
        Get the combination list
        : image_list: a subset of image factors [image1, image2, ...]
        """
        # number of the factors
        org_img_num = len(image_list)
        index = list(range(org_img_num))
        
        combination_list = []
        
        for comb_num in range(2, k + 1):
            # Combination
            sub_index_list = list(combinations(index, comb_num))
            for sub_index in sub_index_list:
                # combinate the masked images
                combination_list.append(sub_index)
        return combination_list
    
    def _combinate_sample(self, image_list, subset_combination_list):
        """
        Get the combination image
        : image_list: a subset of image factors [image1, image2, ...]
        : subset_combination_list: the selected subset index [(0, 1), (0, 2), (0, 3), (0, 4), ...]
        """
        combination_image_list = []
        
        for sub_index in subset_combination_list:
            # combinate the masked images
            combination_mask_image = np.zeros_like(image_list[0])
            for idx in sub_index:
                combination_mask_image = combination_mask_image + image_list[idx]
            combination_image_list.append(combination_mask_image)

        return combination_image_list
    
    def __call__(self, components_image_list, source_image, image_proccess, batch_size = 50):
        """
        Compute Source Face Submodular Score
            @components_image_list: original image (cv2 format) [mask_image 1, ..., mask_image n]
            @source_image: source face image after preproccessing. np.Size([1,3,w,h])
            @image_proccess: a function that pre-proccessing the combination masked image.
        """
        combination_list = self._get_combination_list(components_image_list, self.combination_number_k)
        
        # input_images = [image_proccess(img) for img in image_list]
        components_images = np.array([image_proccess(img) for img in components_image_list])

        u_ = []
        r_ = []
        mc_ = []
        
        with torch.no_grad():
            # Compute the original face feature / 计算原始人脸特征
            source_face_feature = self.face_recognition_model(torch.from_numpy(source_image).to(self.device), remove_head = True)
            # Compute the subset image feature
            components_image_feature = self.face_recognition_model(torch.from_numpy(components_images).to(self.device), remove_head = True)
            
            for step in tqdm(range(math.ceil(len(combination_list) / batch_size))):
                start = step * batch_size
                end = start + batch_size
                
                # batch combination index / 批量索引
                batch_combination_list = combination_list[start:end]
                
                # batch input images / 批量图片
                batch_input_images = self._combinate_sample(components_image_list, batch_combination_list)
                batch_input_images = torch.from_numpy(np.array([image_proccess(img) for img in batch_input_images])).to(self.device)
                
                # Compute uncertainty / 计算不确定性
                u = 1 - self.compute_uncertainty(batch_input_images)
                u_ += u.cpu().numpy().tolist()
                
                # Compute redudancy score / 计算累赘分数
                r = self.proccess_compute_repudancy_score(components_image_feature, batch_combination_list)
                r_ += r.tolist()
                
                # Compute mean closeness score / 计算与原始人脸的相似度
                face_feature = self.face_recognition_model(batch_input_images, remove_head = True)
                mc = self.compute_mean_closeness_score(face_feature, source_face_feature)
                mc_ += mc.cpu().numpy().tolist()
                
                # Face recognition result / 人脸识别的结果，观察是否识别准确
                face_recognition_result = self.face_recognition_model(batch_input_images.to(self.device))
                if step == 0:
                    fr_r = face_recognition_result.cpu().numpy()
                else:
                    fr_r = np.concatenate((fr_r, face_recognition_result.cpu().numpy()), axis=0)
        
            u_ = np.array(u_)
            r_ = np.array(r_)
            mc_ = np.array(mc_)
        
        # print(u, r, mc)
        smdl_score = self.lambda1 * u_ + self.lambda2 * r_ + self.lambda3 * mc_
        return smdl_score, u_, r_, mc_, fr_r, combination_list
    
    # def __call__(self, components_image_list, source_image, image_proccess, batch_size = 50):
    #     """
    #     Compute Source Face Submodular Score
    #         @components_image_list: original image (cv2 format) [mask_image 1, ..., mask_image n]
    #         @source_image: source face image after preproccessing. np.Size([1,3,w,h])
    #         @image_proccess: a function that pre-proccessing the combination masked image.
    #     """
    #     image_list, combination_list = self.combination_mask(components_image_list, self.combination_number_k)
        
    #     input_images = [image_proccess(img) for img in image_list]
        
    #     components_images = np.array([image_proccess(img) for img in components_image_list])

    #     u_ = []
    #     r_ = []
    #     mc_ = []
        
    #     with torch.no_grad():
    #         source_face_feature = self.face_recognition_model(torch.from_numpy(source_image).to(self.device), remove_head = True)
            
    #         components_image_feature = self.face_recognition_model(torch.from_numpy(components_images).to(self.device), remove_head = True)
        
    #     r_ = self.proccess_compute_repudancy_score(components_image_feature, combination_list)
        
    #     for step in range(math.ceil(len(input_images) / batch_size)):
    #         start = step * batch_size
    #         end = start + batch_size
    #         input_images_batch = torch.from_numpy(np.array(input_images[start:end])).to(self.device)

    #         u = 1 - self.compute_uncertainty(input_images_batch.to(self.device))
    #         with torch.no_grad():
    #             face_feature = self.face_recognition_model(input_images_batch.to(self.device), remove_head = True)

    #         mc = self.compute_mean_closeness_score(face_feature, source_face_feature)
            
    #         u_ += u.cpu().numpy().tolist()
    #         mc_ += mc.cpu().numpy().tolist()
            
    #         with torch.no_grad():
    #             face_recognition_result = self.face_recognition_model(input_images_batch.to(self.device))

    #         if step == 0:
    #             fr_r = face_recognition_result.cpu().numpy()
    #         else:
    #             fr_r = np.concatenate((fr_r, face_recognition_result.cpu().numpy()), axis=0)
        
    #     u_ = np.array(u_)
    #     r_ = np.array(r_)
    #     mc_ = np.array(mc_)
        
    #     # print(u, r, mc)
    #     smdl_score = self.lambda1 * u_ + self.lambda2 * r_ + self.lambda3 * mc_
    #     return smdl_score, u_, r_, mc_, fr_r, image_list, combination_list
    
    