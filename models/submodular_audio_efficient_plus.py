import math
import random
import numpy as np

from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from .submodular_vit_torch import MultiModalSubModularExplanation

class AudioSubModularExplanationEfficientPlus(MultiModalSubModularExplanation):
    def __init__(self, 
                 model,
                 semantic_feature,
                 preproccessing_function,
                 k = 40,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda3 = 1.0,
                 lambda4 = 1.0,
                 device = "cuda",
                 pending_samples = 8):
        super(AudioSubModularExplanationEfficientPlus, self).__init__(
            k = k,
            model = model,
            semantic_feature = semantic_feature,
            preproccessing_function = preproccessing_function,
            
            lambda1 = lambda1,
            lambda2 = lambda2,
            lambda3 = lambda3,
            lambda4 = lambda4,
            
            device = device)
        
        # Parameters of the submodular
        self.pending_samples = pending_samples
    
    def merge_image(self, sub_index_set, partition_image_set):
        """
        merge image
        """
        sub_image_set_ = np.array(partition_image_set)[sub_index_set]
        image = sub_image_set_.sum(0)

        return image
    
    def evaluation_maximun_sample(self, 
                                  main_set, 
                                  decrease_set,
                                  candidate_set, 
                                  partition_image_set):
        """
        Given a subset, return a best sample index
        """
        sub_index_sets = []
        for candidate_ in candidate_set:
            sub_index_sets.append(
                np.concatenate((main_set, np.array([candidate_]))).astype(int))
            

        sub_index_sets_decrease = []
        for candidate_ in candidate_set:
            sub_index_sets_decrease.append(
                np.concatenate((decrease_set, np.array([candidate_]))).astype(int))

        # merge images / 组合图像
        sub_images = torch.stack([
            self.preproccessing_function(
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        
        batch_input_images = sub_images.to(self.device)
        
        with torch.no_grad():
            # 2. Effectiveness Score
            score_effectiveness = self.proccess_compute_effectiveness_score(sub_index_sets)
            score_effectiveness_decrease = self.proccess_compute_effectiveness_score(sub_index_sets_decrease)
            
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
            
            # submodular score
            # smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness +  self.lambda3 * score_consistency + self.lambda4 * score_collaboration
            smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness + self.lambda3 * score_consistency + self.lambda4 * score_collaboration
            arg_max_index = smdl_score.argmax().cpu().item()
            
            # if self.lambda1 != 0:
            self.saved_json_file["confidence_score_increase"].append(score_confidence[arg_max_index].cpu().item())
            self.saved_json_file["effectiveness_score_increase"].append(score_effectiveness[arg_max_index].cpu().item())
            self.saved_json_file["consistency_score_increase"].append(score_consistency[arg_max_index].cpu().item())
            self.saved_json_file["collaboration_score_increase"].append(score_collaboration[arg_max_index].cpu().item())
            self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].cpu().item())

            if len(candidate_set) > self.pending_samples:
                smdl_score_decrease = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness_decrease + self.lambda3 * score_consistency + self.lambda4 * score_collaboration
                
                # Select the sample with the worst score as the negative sample estimate
                negtive_sampels_indexes = smdl_score_decrease.topk(self.pending_samples, largest = False).indices.cpu().numpy()
                
                if arg_max_index in negtive_sampels_indexes:
                    negtive_sampels_indexes = negtive_sampels_indexes.tolist()
                    negtive_sampels_indexes.remove(arg_max_index)
                    negtive_sampels_indexes = np.array(negtive_sampels_indexes)
                
                sub_index_negtive_sets = np.array(sub_index_sets_decrease)[negtive_sampels_indexes]
                
                # merge images / 组合图像
                sub_images_decrease = torch.stack([
                    self.preproccessing_function(
                        self.merge_image(sub_index_set, partition_image_set)
                    ) for sub_index_set in sub_index_negtive_sets])
                
                sub_images_decrease_reverse = torch.stack([
                    self.preproccessing_function(
                        self.org_img - self.merge_image(sub_index_set, partition_image_set)
                    ) for sub_index_set in sub_index_negtive_sets])
                
                # 2. Effectiveness Score
                score_effectiveness_decrease_ = score_effectiveness_decrease[negtive_sampels_indexes]
                
                # 3. Consistency Score
                score_consistency_decrease = self.proccess_compute_consistency_score(sub_images_decrease.to(self.device))
                
                # 1. Confidence Score
                score_confidence_decrease = self.proccess_compute_confidence_score()
                
                # 4. Collaboration Score
                score_collaboration_decrease = 1 - self.proccess_compute_consistency_score(sub_images_decrease_reverse.to(self.device))
                
                smdl_score_decrease = self.lambda1 * score_confidence_decrease + self.lambda2 * score_effectiveness_decrease_ + self.lambda3 * score_consistency_decrease + self.lambda4 * score_collaboration_decrease
                arg_min_index = smdl_score_decrease.argmin().cpu().item()
                
                decrease_set = sub_index_negtive_sets[arg_min_index]

                self.saved_json_file["confidence_score_decrease"].append(score_confidence_decrease[arg_min_index].cpu().item())
                self.saved_json_file["effectiveness_score_decrease"].append(score_effectiveness_decrease_[arg_min_index].cpu().item())
                self.saved_json_file["consistency_score_decrease"].append(1-score_collaboration_decrease[arg_min_index].cpu().item())
                self.saved_json_file["collaboration_score_decrease"].append(1-score_consistency_decrease[arg_min_index].cpu().item())

        return sub_index_sets[arg_max_index], decrease_set
    
    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["sub-k"] = self.k
        
        self.saved_json_file["confidence_score"] = []
        self.saved_json_file["effectiveness_score"] = []
        self.saved_json_file["consistency_score"] = []
        self.saved_json_file["collaboration_score"] = []
        
        self.saved_json_file["confidence_score_increase"] = []
        self.saved_json_file["effectiveness_score_increase"] = []
        self.saved_json_file["consistency_score_increase"] = []
        self.saved_json_file["collaboration_score_increase"] = []
        
        self.saved_json_file["confidence_score_decrease"] = []
        self.saved_json_file["effectiveness_score_decrease"] = []
        self.saved_json_file["consistency_score_decrease"] = []
        self.saved_json_file["collaboration_score_decrease"] = []
        
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["lambda3"] = self.lambda3
        self.saved_json_file["lambda4"] = self.lambda4
        
    def get_merge_set(self, partition):
        """
        """
        Subset = np.array([])
        Subset_decrease = np.array([])
        
        indexes = np.arange(len(partition))
        
        # First calculate the similarity of each element to facilitate calculation of effectiveness score.
        self.calculate_distance_of_each_element(partition)
        
        self.smdl_score_best = 0
        
        loop_times = int((self.k-self.pending_samples)/2) + self.pending_samples
        for j in tqdm(range(loop_times)):
            diff = np.setdiff1d(indexes, np.concatenate((Subset, Subset_decrease)))  # in indexes but not in Subset
            
            sub_candidate_indexes = diff
            if len(diff) == 1:
                Subset = np.concatenate((Subset, np.array(diff)))
                break
            
            Subset, Subset_decrease = self.evaluation_maximun_sample(Subset, Subset_decrease, sub_candidate_indexes, partition)
        
        sub_images = torch.stack([
            self.preproccessing_function(
                self.org_img
            ),
            self.preproccessing_function(
                self.org_img - self.org_img
            ),
        ])
        scores = self.proccess_compute_consistency_score(sub_images.to(self.device))
        
        self.saved_json_file["org_score"] = scores[0].cpu().item()
        self.saved_json_file["baseline_score"] = scores[1].cpu().item()
        
        self.saved_json_file["consistency_score"] = self.saved_json_file["consistency_score_increase"] + self.saved_json_file["consistency_score_decrease"][::-1] + [scores[0].cpu().item()]
        self.saved_json_file["collaboration_score"] = self.saved_json_file["collaboration_score_increase"] + self.saved_json_file["collaboration_score_decrease"][::-1] + [1-scores[1].cpu().item()]
        
        Subset = np.concatenate((Subset, Subset_decrease[::-1]))
        
        return Subset.astype(int)
    
    def __call__(self, image_set, id = None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        # V_partition = self.partition_collection(image_set)  # [ [image1, image2, ...], [image1, image2, ...], ...  ]
    
        self.save_file_init()
        
        self.org_img = np.array(image_set).sum(0)
        source_image = self.preproccessing_function(self.org_img)

        self.source_feature = self.model(source_image.unsqueeze(0).to(self.device))
        self.target_label = id
        
        Subset_merge = np.array(image_set)
        Submodular_Subset = self.get_merge_set(Subset_merge)  # array([17, 42, 49, ...])
            
        submodular_image_set = Subset_merge[Submodular_Subset]  # sub_k x (112, 112, 3)
        
        submodular_image = submodular_image_set.sum(0)
        self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
        self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])

        return submodular_image, submodular_image_set, self.saved_json_file