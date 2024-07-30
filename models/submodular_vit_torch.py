import math
import random
import numpy as np

from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
# import torchvision.transforms as transforms

from itertools import combinations
from collections import OrderedDict

class MultiModalSubModularExplanation(object):
    def __init__(self, 
                 model,
                 semantic_feature,
                 preproccessing_function,
                 k = 40,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda3 = 1.0,
                 lambda4 = 1.0,
                 device = "cuda"):
        super(MultiModalSubModularExplanation, self).__init__()
        
        # Parameters of the submodular
        self.k = k
        
        self.model = model
        self.semantic_feature = semantic_feature
        self.preproccessing_function = preproccessing_function
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        
        self.device = device
        
    def partition_collection(self, image_set):
        """
        Divide m image elements into n sets
        """
        image_set_size = len(image_set)
        sample_size_per_partition = image_set_size
        
        image_set_clone = list(image_set)
        random.shuffle(image_set_clone)
        
        V_partition = [image_set_clone[i: i + sample_size_per_partition] for i in range(0, image_set_size, sample_size_per_partition)]
        
        assert len(V_partition[0]) == sample_size_per_partition
        
        self.s_size = sample_size_per_partition
        # assert image_set_size > sample_size_per_partition * self.k  # 其实就是 self.n > self.k ?
        return V_partition
    
    def merge_image(self, sub_index_set, partition_image_set):
        """
        merge image
        """
        sub_image_set_ = np.array(partition_image_set)[sub_index_set]
        image = sub_image_set_.sum(0)

        return image.astype(np.uint8)
    
    # def compute_effectiveness_score(self, features):
    #     """
    #     Computes Eeffectiveness Score: The point should be distant from all the other elements in the subset.
    #     features: torch.Size(batch, d)
    #     """
    #     norm_feature = F.normalize(features, p=2, dim=1)
    #     # Consine Similarity
    #     cosine_similarity = torch.mm(norm_feature, norm_feature.t())
    #     cosine_similarity = torch.clamp(cosine_similarity, min=-1, max=1)
    #     # Normlize 0-1
    #     cosine_dist = torch.arccos(cosine_similarity) / math.pi
        
    #     if cosine_dist.shape[0] == 1:
    #         eye = 1 - torch.eye(norm_feature.shape[0], device=self.device)
    #         masked_dist = cosine_dist * eye
    #         e_score = torch.sum(torch.min(masked_dist, dim=1).values)
    #     else:
    #         eye = torch.eye(norm_feature.shape[0], device=self.device)
    #         adjusted_cosine_dist = cosine_dist + eye
    #         e_score = torch.sum(torch.min(adjusted_cosine_dist, dim=1).values)
        
    #     return e_score # tensor(0.0343, device='cuda:0')
        
    # def proccess_compute_effectiveness_score_v1(self, components_image_feature, combination_list):
    #     """
    #     Compute each S's effectiveness score
    #     """
    #     e_scores = []
    #     for sub_index in combination_list:
    #         sub_feature_set = components_image_feature[sub_index]
    #         e_score = self.compute_effectiveness_score(sub_feature_set)
    #         e_scores.append(e_score)
        
    #     return torch.stack(e_scores)
    
    def proccess_compute_confidence_score(self):
        """
        Compute confidence score
        """
        # visual_features = self.model(batch_input_images)
        # predicted_scores = torch.softmax(visual_features @ self.semantic_feature.T, dim=-1)
        entropy = - torch.sum(self.predicted_scores * torch.log(self.predicted_scores + 1e-7), dim=1)
        max_entropy = torch.log(torch.tensor(self.predicted_scores.shape[1])).to(self.device)
        confidence = 1 - (entropy / max_entropy)
        return confidence 
    
    def proccess_compute_effectiveness_score(self, sub_index_sets):
        """
        Compute each S's effectiveness score
        """
        e_scores = []
        
        for sub_index in sub_index_sets:
            cosine_dist = self.effectiveness_dist[:, np.array(sub_index)]    # [len(element) , len(main_set)]
            cosine_dist = cosine_dist[np.array(sub_index), :]
            
            eye = torch.eye(cosine_dist.shape[0], device=self.device)
            adjusted_cosine_dist = cosine_dist + eye
            e_score = torch.sum(torch.min(adjusted_cosine_dist, dim=1).values)
            e_scores.append(e_score)
        
        effectiveness_score = torch.stack(e_scores)
        if len(sub_index_sets[0]) == 1:
            effectiveness_score = effectiveness_score * 0
        return effectiveness_score
    
    def proccess_compute_consistency_score(self, batch_input_images):
        """
        Compute each consistency score
        """
        with torch.no_grad():
            visual_features = self.model(batch_input_images)
            self.predicted_scores = torch.softmax(visual_features @ self.semantic_feature.T, dim=-1)
            consistency_scores = self.predicted_scores[:, self.target_label]

        return consistency_scores
    
    def evaluation_maximun_sample(self, 
                                  main_set, 
                                  candidate_set, 
                                  partition_image_set):
        """
        Given a subset, return a best sample index
        """
        sub_index_sets = []
        for candidate_ in candidate_set:
            sub_index_sets.append(
                np.concatenate((main_set, np.array([candidate_]))).astype(int))
       
        # merge images / 组合图像
        sub_images = torch.stack([
            self.preproccessing_function(
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        
        batch_input_images = sub_images.to(self.device)
        
        with torch.no_grad():
            
            # 2. Effectiveness Score
            score_effectiveness = self.proccess_compute_effectiveness_score(sub_index_sets)
        
            # 3. Consistency Score
            score_consistency = self.proccess_compute_consistency_score(batch_input_images)
            
            # 1. Confidence Score
            score_confidence = self.proccess_compute_confidence_score()
            
            # 4. Collaboration Score
            sub_images_reverse = torch.stack([
                self.preproccessing_function(
                    self.org_img - self.merge_image(sub_index_set, partition_image_set)
                ) for sub_index_set in sub_index_sets])
        
            batch_input_images_reverse = sub_images_reverse.to(self.device)
            
            score_collaboration = 1 - self.proccess_compute_consistency_score(batch_input_images_reverse)
            
            # 1. Confidence Score
            # score_confidence = self.proccess_compute_confidence_score()
            
            # submodular score
            smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness +  self.lambda3 * score_consistency + self.lambda4 * score_collaboration
            # smdl_score = self.lambda2 * score_effectiveness +  self.lambda3 * score_consistency + self.lambda4 * score_collaboration
            arg_max_index = smdl_score.argmax().cpu().item()
            
            # if self.lambda1 != 0:
            self.saved_json_file["confidence_score"].append(score_confidence[arg_max_index].cpu().item())
            self.saved_json_file["effectiveness_score"].append(score_effectiveness[arg_max_index].cpu().item())
            self.saved_json_file["consistency_score"].append(score_consistency[arg_max_index].cpu().item())
            self.saved_json_file["collaboration_score"].append(score_collaboration[arg_max_index].cpu().item())
            self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].cpu().item())
        
        return sub_index_sets[arg_max_index]
    
    def save_file_init(self):
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
    
    def calculate_distance_of_each_element(self, partition_image_set):
        """
        Calculate the similarity of each element, obtain a similarity matrix
        """
        with torch.no_grad():
            partition_images = torch.stack([
                self.preproccessing_function(
                    partition_image
                ) for partition_image in partition_image_set]).to(self.device)
            partition_image_features = self.model(partition_images)
            
            norm_feature = F.normalize(partition_image_features, p=2, dim=1)
            # Consine Similarity
            cosine_similarity = torch.mm(norm_feature, norm_feature.t())
            cosine_similarity = torch.clamp(cosine_similarity, min=-1, max=1)
            
            # Normlize 0-1
            self.effectiveness_dist = torch.arccos(cosine_similarity) / math.pi
    
    def get_merge_set(self, partition):
        """
        """
        Subset = np.array([])
        
        indexes = np.arange(len(partition))
        
        # First calculate the similarity of each element to facilitate calculation of effectiveness score.
        self.calculate_distance_of_each_element(partition)
        
        self.smdl_score_best = 0
        
        for j in tqdm(range(self.k)):
            diff = np.setdiff1d(indexes, np.array(Subset))  # in indexes but not in Subset
            
            sub_candidate_indexes = diff
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition)
        
        return Subset
    
    def __call__(self, image_set, id = None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        # V_partition = self.partition_collection(image_set)  # [ [image1, image2, ...], [image1, image2, ...], ...  ]
    
        self.save_file_init()
        
        self.org_img = np.array(image_set).sum(0).astype(np.uint8)      
        source_image = self.preproccessing_function(self.org_img)

        self.source_feature = self.model(source_image.unsqueeze(0).to(self.device))
        self.target_label = id
        
        Subset_merge = np.array(image_set)
        Submodular_Subset = self.get_merge_set(Subset_merge)  # array([17, 42, 49, ...])
            
        submodular_image_set = Subset_merge[Submodular_Subset]  # sub_k x (112, 112, 3)
        
        
        submodular_image = submodular_image_set.sum(0).astype(np.uint8)
        self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
        self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
        return submodular_image, submodular_image_set, self.saved_json_file