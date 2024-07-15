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
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image
import torchvision.transforms as transforms

from itertools import combinations
from collections import OrderedDict

from .iresnet import iresnet50
from .iresnet_edl import iresnet100
from .evidential import relu_evidence, exp_evidence

import tensorflow_addons as tfa
from keras.models import load_model
from insight_face_models import *

class FaceSubModularExplanation(object):
    def __init__(self, 
                 cfg_path="configs/celeba/submodular_cfg_celeba_tf.json",
                 n = 2,
                 k = 40,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda3 = 1.0,
                 lambda4 = 1.0):
        super(FaceSubModularExplanation, self).__init__()
        
        # Load face model configuration / 导入人脸识别模型的配置文件
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        
        self.device = torch.device(self.cfg["device"])
        self.moda = self.cfg["mode"]
            
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        # Uncertainty estimation model / 不确定性估计模型
        self.uncertainty_model = self.define_uncertrainty_network(
            self.cfg["uncertainty_model"]["num_classes"], self.cfg["uncertainty_model"]["model_path"])
        # Face recognition
        self.face_recognition_model = self.define_recognition_model(
            self.cfg["face_recognition_model"]["num_classes"], self.cfg["face_recognition_model"]["model_path"])

        # Parameters of the submodular / submodular的超参数
        self.n = n  # the number of the partitions / 图像元素集被划分的数量
        self.k = k
        
        # Parameter of the LtLG algorithm / LtLG贪婪算法的参数
        self.ltl_log_ep = 5
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        if self.moda == "Torch":
            self.softmax = torch.nn.Softmax(dim=1)
        elif self.moda == "TF":
            self.softmax = tf.keras.layers.Softmax(axis=-1)

    def convert_prepare_image(self, image, size=112, moda="Torch"):
        if moda == "Torch":
            img = cv2.resize(image, (size, size))
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transforms(img).numpy()
        elif moda == "TF":
            img = cv2.resize(image[...,::-1], (size, size))
            img = (img - 127.5) * 0.0078125
            img = img.astype(np.float32)
        # img = img.transpose(1, 2, 0)
        return img

    def define_recognition_model(self, num_classes, pretrained_path):
        """
        init the face recognition model
        """
        if self.moda == "Torch":
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
        
        elif self.moda == "TF":
            self.model_base = load_model(pretrained_path)
            model = tf.keras.models.Model(inputs=self.model_base.input, outputs=self.model_base.get_layer("embedding").output)
            print("Success load pre-trained face recognition model {}".format(pretrained_path))

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
            consine_dist = 1 - torch.arccos(consine_similarity) / math.pi   # Is distance, not similarity! no need 1-. Bug need revision

            mc_score = consine_dist.reshape(-1)
            # mc_score = consine_similarity.reshape(-1)
        return mc_score
    
    def partition_collection(self, image_set):
        """
        Divide m image elements into n sets
        """
        image_set_size = len(image_set)
        sample_size_per_partition = int(image_set_size / self.n)
        
        image_set_clone = list(image_set)
        random.shuffle(image_set_clone)
        
        V_partition = [image_set_clone[i: i + sample_size_per_partition] for i in range(0, image_set_size, sample_size_per_partition)]
        
        assert len(V_partition) == self.n
        assert len(V_partition[0]) == sample_size_per_partition
        
        self.s_size = int((sample_size_per_partition * self.ltl_log_ep) / self.n)
        # assert image_set_size > sample_size_per_partition * self.k  # 其实就是 self.n > self.k ?
        return V_partition
    
    def merge_image(self, sub_index_set, partition_image_set):
        """
        merge image
        """
        sub_image_set_ = np.array(partition_image_set)[sub_index_set]
        image = sub_image_set_.sum(0)

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
        
        # merge images / 组合图像
        sub_images = np.array([
            self.convert_prepare_image(
                self.merge_image(sub_index_set, partition_image_set), moda="Torch"  # Uncertainty model is pytorch version
            ) for sub_index_set in sub_index_sets])
        
        batch_input_images = torch.from_numpy(sub_images).type(torch.float32).to(self.device)
        
        with torch.no_grad():
            if self.lambda1 == 0:
                confidence = 0
            else:
                # Compute uncertainty / 计算不确定性
                u = self.compute_uncertainty(
                    batch_input_images
                )
                confidence = 1 - u
            
            if self.lambda2 == 0:
                r = 0
            else:
                r = self.process_compute_effectiveness_score(sub_index_sets)
            
            # Compute mean closeness score / 计算与原始人脸的相似度 (越相似越好吧)
            batch_input_images = np.array([
                self.convert_prepare_image(
                    self.merge_image(sub_index_set, partition_image_set), moda=self.moda  # Uncertainty model is pytorch version
                ) for sub_index_set in sub_index_sets])
            
            batch_input_images_reverse = np.array([
                self.convert_prepare_image(
                    self.org_img - self.merge_image(sub_index_set, partition_image_set), moda=self.moda  # Uncertainty model is pytorch version
                ) for sub_index_set in sub_index_sets])
            
            consistency_score = self.model_base(batch_input_images)[:, self.target_label].numpy()
            score_collaboration = 1 - self.model_base(batch_input_images_reverse)[:, self.target_label].numpy()
                
        smdl_score = self.lambda1 * confidence.cpu().numpy() + self.lambda2 * r +  self.lambda3 * consistency_score + self.lambda4 * score_collaboration
        
        if not monotonically_increasing:
            arg_max_index = smdl_score.argmax()
            return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
        
        arg_max_index = smdl_score.argmax()
        if self.lambda1 != 0:
            self.saved_json_file["confidence_score"].append(confidence[arg_max_index].cpu().item())
        if self.lambda2 != 0:
            self.saved_json_file["effectiveness_score"].append(r[arg_max_index].item())
        if self.lambda3 != 0:
            self.saved_json_file["consistency_score"].append(consistency_score[arg_max_index].item())
        if self.lambda4 != 0:
            self.saved_json_file["collaboration_score"].append(score_collaboration[arg_max_index].item())
        self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].item())
        
        return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
    
    def calculate_distance_of_each_element(self, partition_image_set):
        partition_image_features = np.array([
            self.convert_prepare_image(
                partition_image, moda=self.moda
            ) for partition_image in partition_image_set
        ])
        
        partition_image_features = self.face_recognition_model(
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

            # sub_candidate_indexes = np.random.choice(diff, self.s_size)
            sub_candidate_indexes = diff
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition, monotonically_increasing)
            
        return Subset
    
    def __call__(self, image_set, target_label=None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        V_partition = self.partition_collection(image_set)  # [ [image1, image2, ...], [image1, image2, ...], ...  ]
        
        self.saved_json_file = {}
        self.saved_json_file["sub-n"] = self.n
        self.saved_json_file["sub-k"] = self.k
        self.saved_json_file["confidence_score"] = []
        self.saved_json_file["effectiveness_score"] = []
        self.saved_json_file["consistency_score"] = []
        self.saved_json_file["collaboration_score"] = []
        self.saved_json_file["smdl_score"] = []
        # self.saved_json_file["recognition_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["lambda3"] = self.lambda3
        self.saved_json_file["lambda4"] = self.lambda4
        
        self.org_img = np.array(image_set).sum(0).astype(np.uint8)
        source_image = self.convert_prepare_image(
                np.array(image_set).sum(0).astype(np.uint8), moda = self.moda)
        
        if self.moda == "Torch":
            self.source_feature = self.face_recognition_model(
                torch.from_numpy(source_image).unsqueeze(0).to(self.device), 
                remove_head = True)
        
            if target_label == None:
                self.target_label = self.face_recognition_model(torch.from_numpy(source_image).unsqueeze(0).to(self.device), remove_head = False).argmax().item()
            else:
                self.target_label = target_label
        elif self.moda == "TF":
            self.source_feature = self.face_recognition_model(np.array([source_image]))
            self.source_feature = torch.from_numpy(
                self.source_feature.numpy()).to(self.device)
            if target_label == None:
                predict = self.model_base(np.array([source_image]))
                self.target_label = predict.numpy().argmax()
            else:
                self.target_label = target_label
        if self.n != 1:
            Subset_merge = []
            for partition in V_partition:
                Subset = self.get_merge_set(partition)  # array([17, 42, 49, ...])
                Subset_merge.append(np.array(partition)[Subset])
            
            Subset_merge = np.concatenate(Subset_merge) # np.shape: (60, 112, 112, 3)
        
        else:
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
    