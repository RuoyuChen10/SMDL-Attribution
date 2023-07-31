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

class FaceSubModularExplanation(object):
    def __init__(self, 
                 cfg_path="models/submodular_cfg.json",
                 n = 2,
                 k = 40):
        super(FaceSubModularExplanation, self).__init__()
        
        # Load face model configuration / 导入人脸识别模型的配置文件
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        
        self.device = torch.device(self.cfg["device"])
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
        
        self.lambda1 = 1
        self.lambda2 = 1
        self.lambda3 = 2
        
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def convert_prepare_image(self, image, size=112):
        img = cv2.resize(image, (size, size))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transforms(img).numpy()
        # img = img.transpose(1, 2, 0)
        return img

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
            
            if consine_dist.shape[0] == 1:
                r_scores = torch.min(consine_dist * (1 - torch.eye(norm_feature.shape[0]).to(self.device)), -1)[0].sum()
            else:
                r_scores = torch.min(
                    consine_dist + torch.eye(norm_feature.shape[0]).to(self.device),
                    -1)[0].sum()    # fixed bug

        return r_scores # tensor(0.0343, device='cuda:0')
    
    def proccess_compute_repudancy_score(self, components_image_feature, combination_list):
        """
        Compute each S's r score
        """
        r_scores = []
        for sub_index in combination_list:
            sub_feature_set = components_image_feature[sub_index]
            r_score = self.compute_redundancy_score(sub_feature_set)
            r_scores.append(r_score.cpu().numpy())
        
        return torch.from_numpy(np.array(r_scores)).to(self.device)
    
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
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        batch_input_images = torch.from_numpy(sub_images).type(torch.float32).to(self.device)
        
        with torch.no_grad():
            # Compute uncertainty / 计算不确定性
            u = self.compute_uncertainty(
                batch_input_images
            )
            confidence = 1 - u
            
            # Compute redudancy score / 计算累赘分数
            partition_image_features = np.array([
                self.convert_prepare_image(
                    partition_image
                ) for partition_image in partition_image_set
            ])
            partition_image_features = self.face_recognition_model(
                torch.from_numpy(partition_image_features).type(torch.float32).to(self.device),
                remove_head = True
            )
            # print(partition_image_features.shape, sub_index_sets)
            r = self.proccess_compute_repudancy_score(partition_image_features, sub_index_sets)
            
            # Compute mean closeness score / 计算与原始人脸的相似度 (越相似越好吧)
            face_feature = self.face_recognition_model(batch_input_images, remove_head = True)
            mc = self.compute_mean_closeness_score(face_feature, self.source_feature)
            
            recognition_score = self.softmax(self.face_recognition_model(batch_input_images, remove_head = False))[:, self.target_label]
            
        smdl_score = self.lambda1 * confidence + self.lambda2 * r +  self.lambda3 * mc
        
        if not monotonically_increasing:
            arg_max_index = smdl_score.argmax()
            return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
        
        # The maximum score
        # if monotonically_increasing:
        #     arg_max_index = smdl_score.argmax()
        #     print((self.lambda1 * confidence)[arg_max_index].cpu().item(),
        #           (self.lambda2 * r)[arg_max_index].cpu().item(),
        #           (self.lambda3 * mc)[arg_max_index].cpu().item(),
        #           )
        
        # if smdl_score.max() > self.smdl_score_best:
            # self.smdl_score_best = smdl_score.max()
        arg_max_index = smdl_score.argmax()
    
        self.saved_json_file["confidence_score"].append(confidence[arg_max_index].cpu().item())
        self.saved_json_file["redundancy_score"].append(r[arg_max_index].cpu().item())
        self.saved_json_file["verification_score"].append(mc[arg_max_index].cpu().item())
        self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].cpu().item())
        self.saved_json_file["recognition_score"].append(recognition_score[arg_max_index].cpu().item())

        return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
    
        # else:
        #     return main_set
    
    def get_merge_set(self, partition, monotonically_increasing = False):
        """
        """
        Subset = np.array([])
        
        indexes = np.arange(len(partition))
        
        self.smdl_score_best = 0
        
        for j in range(self.k):
            # Sample a subsize of size s.
            diff = np.setdiff1d(indexes, np.array(Subset))  # in indexes but not in Subset

            sub_candidate_indexes = np.random.choice(diff, self.s_size)
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition, monotonically_increasing)
            
        return Subset
    
    def __call__(self, image_set):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        V_partition = self.partition_collection(image_set)  # [ [image1, image2, ...], [image1, image2, ...], ...  ]
        
        self.saved_json_file = {}
        self.saved_json_file["sub-n"] = self.n
        self.saved_json_file["sub-k"] = self.k
        self.saved_json_file["confidence_score"] = []
        self.saved_json_file["redundancy_score"] = []
        self.saved_json_file["verification_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["recognition_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["lambda3"] = self.lambda3
        
        source_image = self.convert_prepare_image(
            np.array(image_set).sum(0).astype(np.uint8))
        self.source_feature = self.face_recognition_model(
            torch.from_numpy(source_image).unsqueeze(0).to(self.device), 
            remove_head = True)
        
        self.target_label = self.face_recognition_model(torch.from_numpy(source_image).unsqueeze(0).to(self.device), remove_head = False).argmax().item()
        
        Subset_merge = []
        
        for partition in V_partition:
            Subset = self.get_merge_set(partition)  # array([17, 42, 49, ...])
            Subset_merge.append(np.array(partition)[Subset])
        
        Subset_merge = np.concatenate(Subset_merge) # np.shape: (60, 112, 112, 3)
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

class FaceSubModularExplanation_v1(object):
    def __init__(self, 
                 cfg_path="models/submodular_cfg.json",
                 n = 2,
                 k = 40):
        super(FaceSubModularExplanation_v1, self).__init__()
        
        # Load face model configuration / 导入人脸识别模型的配置文件
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        
        self.device = torch.device(self.cfg["device"])
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
        
        self.lambda1 = 1
        self.lambda2 = 10
        self.lambda3 = 1
        
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
    def convert_prepare_image(self, image, size=112):
        img = cv2.resize(image, (size, size))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transforms(img).numpy()
        # img = img.transpose(1, 2, 0)
        return img
        
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
            
            if consine_dist.shape[0] == 1:
                r_scores = torch.min(consine_dist * (1 - torch.eye(norm_feature.shape[0]).to(self.device)), -1)[0].min()
            else:
                r_scores = torch.min(
                    consine_dist + torch.eye(norm_feature.shape[0]).to(self.device),
                    -1)[0].min()

        return r_scores # tensor(0.0343, device='cuda:0')
    
    def proccess_compute_repudancy_score(self, components_image_feature, combination_list):
        """
        Compute each S's r score
        """
        r_scores = []
        for sub_index in combination_list:
            sub_feature_set = components_image_feature[sub_index]
            r_score = self.compute_redundancy_score(sub_feature_set)
            r_scores.append(r_score.cpu().numpy())
        
        return torch.from_numpy(np.array(r_scores)).to(self.device)
    
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
    
    def evaluation_maximun_sample(self, main_set, candidate_set, partition_image_set):
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
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        batch_input_images = torch.from_numpy(sub_images).type(torch.float32).to(self.device)
        
        with torch.no_grad():
            # Compute uncertainty / 计算不确定性
            u = self.compute_uncertainty(
                batch_input_images
            )
            confidence = 1 - u
            
            # Compute redudancy score / 计算累赘分数
            partition_image_features = np.array([
                self.convert_prepare_image(
                    partition_image
                ) for partition_image in partition_image_set
            ])
            partition_image_features = self.face_recognition_model(
                torch.from_numpy(partition_image_features).type(torch.float32).to(self.device),
                remove_head = True
            )
            # print(partition_image_features.shape, sub_index_sets)
            r = self.proccess_compute_repudancy_score(partition_image_features, sub_index_sets)
            
            # Compute mean closeness score / 计算与原始人脸的相似度 (越相似越好吧)
            face_feature = self.face_recognition_model(batch_input_images, remove_head = True)
            mc = self.compute_mean_closeness_score(face_feature, self.source_feature)
            
        smdl_score = self.lambda1 * confidence + self.lambda2 * r +  self.lambda3 * mc
        # The maximum score
        # if smdl_score.max() > self.smdl_score_best:
        self.smdl_score_best = smdl_score.max()
        arg_max_index = smdl_score.argmax()
    
        return sub_index_sets[arg_max_index]
    
        # else:
        #     return main_set
    
    def get_merge_set(self, partition):
        """
        """
        Subset = np.array([])
        
        indexes = np.arange(len(partition))
        
        self.smdl_score_best = 0
        
        for j in range(self.k):
            # Sample a subsize of size s.
            diff = np.setdiff1d(indexes, np.array(Subset))

            sub_candidate_indexes = np.random.choice(diff, self.s_size)
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition)
            
        return Subset
    
    def __call__(self, image_set):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        V_partition = self.partition_collection(image_set)
        
        source_image = self.convert_prepare_image(
            np.array(image_set).sum(0).astype(np.uint8))
        self.source_feature = self.face_recognition_model(
            torch.from_numpy(source_image).unsqueeze(0).to(self.device), 
            remove_head = True)
        
        Subset_merge = []
        
        for partition in V_partition:
            Subset = self.get_merge_set(partition)            
            Subset_merge.append(np.array(partition)[Subset])
        
        Subset_merge = np.concatenate(Subset_merge)
        Submodular_Subset = self.get_merge_set(Subset_merge)     

        submodular_image_set = Subset_merge[Submodular_Subset]
        submodular_image = submodular_image_set.sum(0).astype(np.uint8)
        
        # cv2.imwrite("submodular.jpg", submodular_image)
        return submodular_image
    