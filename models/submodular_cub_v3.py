# -*- coding: utf-8 -*-  

"""
Created on 2023/5/18

@author: Ruoyu Chen
"""

import json
import os
import torch
import math
import random
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf

from collections import OrderedDict

import tensorflow_addons as tfa
from keras.models import load_model
from insight_face_models import *

import time

import torchvision.transforms as transforms
from .evidential import relu_evidence, exp_evidence

from tqdm import tqdm

class CubSubModularExplanationV3(object):
    def __init__(self, 
                 cfg_path="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
                 k = 40,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda3 = 1.0,
                 lambda4 = 1.0):
        super(CubSubModularExplanationV3, self).__init__()
        
        # Load model configuration / 导入模型的配置文件
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        assert self.cfg["version"] == 2
        
        self.device = torch.device(self.cfg["device"])
        self.moda = self.cfg["mode"]

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
            
        # Uncertainty estimation model / 不确定性估计模型
        self.uncertainty_model = self.define_uncertrainty_network(
            self.cfg["uncertainty_model"]["model_path"])
        # Face recognition
        self.recognition_model = self.define_recognition_model(
            self.cfg["recognition_model"]["num_classes"], self.cfg["recognition_model"]["model_path"])

        # Parameters of the submodular / submodular的超参数
        self.k = k
        
        # Parameter of the LtLG algorithm / LtLG贪婪算法的参数
        self.ltl_log_ep = 5
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        # self.softmax = tf.keras.layers.Softmax(axis=-1)

        if "resnet" in self.cfg["recognition_model"]["model_path"]:
            from keras.applications.resnet import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224
        elif "vgg19" in self.cfg["recognition_model"]["model_path"]:
            from keras.applications.vgg19 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224
        elif "efficientnetv2" in self.cfg["recognition_model"]["model_path"]:
            from keras.applications.efficientnet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 384
        elif "mobilenetv2" in self.cfg["recognition_model"]["model_path"]:
            print("mobilenetv2")
            from keras.applications.mobilenet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224

    def convert_prepare_image(self, image):
        img = cv2.resize(image[...,::-1], (self.tf_size, self.tf_size))
        img = self.preprocess_input(np.array(img))
        
        return img
    
    def preprocess_image_uncertainty(self, image, size=224):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = self.transforms(img).numpy()
        return img

    def define_recognition_model(self, num_classes, pretrained_path):
        """
        init the face recognition model
        """
        model_base = load_model(pretrained_path)
        layer_name = "dense"
        # layer_name = "flatten"
        model = tf.keras.models.Model(inputs=model_base.input, outputs=[model_base.get_layer(layer_name).output, model_base.output])
        # model.layers[-1].activation = tf.keras.activations.linear
        print("Success load pre-trained bird recognition model {}".format(pretrained_path))

        return model

    def define_uncertrainty_network(self, pretrained_path):
        """
        Init the uncertainty model
        """
        from torchvision import models
        import torch.nn as nn
        model=models.resnet101(pretrained=False) # resnet152
        channel_in = model.fc.in_features
        model.fc = nn.Linear(channel_in, self.cfg["uncertainty_model"]["num_classes"])

        if pretrained_path is not None and os.path.exists(pretrained_path):
            model_dict = model.state_dict()
            pretrained_param = torch.load(pretrained_path, map_location=torch.device('cpu'))

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
    
    def exp_evidence(self, y):
        # 使用np.clip限制y的值在-10到10之间，然后计算指数
        return np.exp(np.clip(y, -10, 10))

    def compute_uncertainty(self, input_images, scale = 5):
        """
        Compute the uncertainty of the model
        input: torch.Size(batch, 3, w, h)
        """
        with torch.no_grad():
            input_images = torch.from_numpy(input_images)
            output = self.uncertainty_model(input_images.to(self.device))
        evidence = exp_evidence(output)
        alpha = evidence + 1
        uncertainty = self.cfg["uncertainty_model"]["num_classes"] / torch.sum(alpha, dim=1, keepdim=True)

        return uncertainty.reshape(-1).cpu().numpy()
    
    # def compute_effectiveness_score(self, features):
    #     """
    #     Computes Eeffectiveness Score: The point should be distant from all the other elements in the subset.
    #     features: torch.Size(batch, d)
    #     """
    #     if self.cfg["effectiveness_distance_metric"] == "cosine":
    #         norm_feature = tf.nn.l2_normalize(features, axis=1)
    #         # Consine Similarity
    #         cosine_similarity = tf.matmul(norm_feature, tf.transpose(norm_feature))
    #         cosine_similarity = tf.clip_by_value(cosine_similarity, -1, 1)
    #         # Normlize 0-1
    #         cosine_dist = tf.acos(cosine_similarity) / math.pi
            
    #         if cosine_dist.shape[0] == 1:
    #             eye = 1 - tf.eye(norm_feature.shape[0])
    #             masked_dist = cosine_dist * eye
    #             e_score = tf.reduce_sum(tf.reduce_min(masked_dist, axis=1))
    #         else:
    #             # e_scores = torch.min(
    #                 # cosine_dist + torch.eye(norm_feature.shape[0]).to(self.device),
    #                 # -1)[0].sum()    # fixed bug
    #             eye = tf.eye(norm_feature.shape[0])
    #             adjusted_cosine_dist = cosine_dist + eye
    #             e_score = tf.reduce_sum(
    #                 tf.reduce_min(adjusted_cosine_dist, axis=1))
    #     return e_score # tensor(0.0343, device='cuda:0')
    
    # def proccess_compute_effectiveness_score(self, components_image_feature, combination_list):
    #     """
    #     Compute each S's effectiveness score
    #     """
    #     e_scores = []
    #     for sub_index in combination_list:
    #         sub_feature_set = tf.gather(components_image_feature, sub_index)    # shape=(batch, 1024)

    #         e_score = self.compute_effectiveness_score(sub_feature_set)
    #         e_scores.append(e_score.numpy())
        
    #     return np.array(e_scores)
    def process_compute_effectiveness_score(self, sub_index_sets):
        """
        Compute each S's effectiveness score
        """
        e_scores = []
        
        for sub_index in sub_index_sets:
            cosine_dist = tf.gather(self.effectiveness_dist, sub_index, axis=1)  # [len(element), len(main_set)]
            cosine_dist = tf.gather(cosine_dist, sub_index, axis=0)
            
            eye = tf.eye(cosine_dist.shape[0])
            adjusted_cosine_dist = cosine_dist + eye
            e_score = tf.reduce_sum(tf.reduce_min(adjusted_cosine_dist, axis=1))
            e_scores.append(e_score)
        
        return tf.stack(e_scores).numpy()
    
    def merge_image(self, sub_index_set, partition_image_set, mode = "black"):
        """
        merge image
        """
        sub_image_set_ = np.array(partition_image_set)[sub_index_set]
        if mode == "black":
            image = sub_image_set_.sum(0)
        elif mode == "gray":
            image = sub_image_set_.sum(0)
            image[image.sum(-1)==0] = 127

        return image.astype(np.uint8)
    
    def evaluation_maximun_sample(self, 
                                  main_set, 
                                  candidate_set, 
                                  partition_image_set, 
                                  monotonically_increasing):
        """
        Given a subset, return a best sample index
        """
        sub_index_sets = []
        for candidate_ in candidate_set:
            sub_index_sets.append(
                np.concatenate((main_set, np.array([candidate_]))).astype(int))

        # Compute uncertainty / 计算不确定性
        start = time.time()
        # merge images / 组合图像
        batch_input_images_u = np.array([
            self.preprocess_image_uncertainty(
                self.merge_image(sub_index_set, partition_image_set)  # Uncertainty model is ONNX version
            ) for sub_index_set in sub_index_sets])
        
        u = self.compute_uncertainty(
            batch_input_images_u
        )
        score_confidence = 1 - u
        end = time.time()
        # print('confidence程序执行时间: ',end - start)
        
        # Compute Effectiveness Score / 计算有效性分数
        start = time.time()
        score_effectiveness = self.process_compute_effectiveness_score(
            sub_index_sets)
        end = time.time()
        # print('effectiveness程序执行时间: ',end - start)
        
        # Compute Consistency Score 
        start = time.time()
        batch_input_images = np.array([
            self.convert_prepare_image(
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        _, score_consistency = self.recognition_model(batch_input_images)
        score_consistency = score_consistency.numpy()[:, self.target_label]
        end = time.time()
        # print('consistency程序执行时间: ',end - start)
        
        # Compute Collaboration Score 
        start = time.time()
        batch_input_images_reverse = np.array([
            self.convert_prepare_image(
                self.org_img - self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        _, score_collaboration = self.recognition_model(batch_input_images_reverse)
        score_collaboration = 1 - score_collaboration.numpy()[:, self.target_label]
        end = time.time()
        # print('collaboration程序执行时间: ',end - start)

        # submodular score
        smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness +  self.lambda3 * score_consistency + self.lambda4 * score_collaboration
        
        arg_max_index = smdl_score.argmax().item()
    
        self.saved_json_file["confidence_score"].append(score_confidence[arg_max_index].item())
        self.saved_json_file["effectiveness_score"].append(score_effectiveness[arg_max_index].item())
        self.saved_json_file["consistency_score"].append(score_consistency[arg_max_index].item())
        self.saved_json_file["collaboration_score"].append(score_collaboration[arg_max_index].item())
        self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].item())

        return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
    
    def calculate_distance_of_each_element(self, partition_image_set):
        partition_image_features = np.array([
            self.convert_prepare_image(
                partition_image
            ) for partition_image in partition_image_set
        ])
        
        partition_image_features, _ = self.recognition_model(
            partition_image_features
        )
        
        norm_feature = tf.nn.l2_normalize(partition_image_features, axis=1)
        # Consine Similarity
        cosine_similarity = tf.matmul(norm_feature, tf.transpose(norm_feature))
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1, 1)
        # Normlize 0-1
        self.effectiveness_dist = tf.acos(cosine_similarity) / math.pi
    
    def get_merge_set(self, partition, monotonically_increasing = False):
        """
        """
        Subset = np.array([])
        
        indexes = np.arange(len(partition))
        
        self.calculate_distance_of_each_element(partition)
        
        self.smdl_score_best = 0
        
        for j in tqdm(range(self.k)):
            # Sample a subsize of size s.
            diff = np.setdiff1d(indexes, np.array(Subset))  # in indexes but not in Subset

            sub_candidate_indexes = diff
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition, monotonically_increasing)
            
        return Subset
    
    def __call__(self, image_set, id = None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        self.saved_json_file = {}
        self.saved_json_file["sub-k"] = self.k
        self.saved_json_file["confidence_score"] = []
        self.saved_json_file["effectiveness_score"] = []
        self.saved_json_file["consistency_score"] = []
        self.saved_json_file["collaboration_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["lambda3"] = self.lambda3
        self.saved_json_file["lambda4"] = self.lambda4
        
        self.org_img = np.array(image_set).sum(0).astype(np.uint8)
        
        if id == None:
            source_image = self.convert_prepare_image(
                self.org_img)
            self.source_feature, predict = self.recognition_model(np.array([source_image]))
            self.target_label = predict.numpy().argmax()
        else:
            self.target_label = id

        Subset_merge = np.array(image_set)
        # print(Subset_merge.shape)
        # cv2.imwrite("Subset_merge.jpg", Subset_merge.sum(0))
        Submodular_Subset = self.get_merge_set(     # array([30, 31,  1, ...])
            Subset_merge, 
            monotonically_increasing=True)

        submodular_image_set = Subset_merge[Submodular_Subset]  # sub_k x (112, 112, 3)
        
        submodular_image = submodular_image_set.sum(0).astype(np.uint8)

        self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
        self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
        return submodular_image, submodular_image_set, self.saved_json_file