import os
import sys
from .cam import get_feature_map
from .causal_score import causal_score
import numpy as np
import cv2
import copy
from skimage.transform import resize
from sklearn.cluster import AgglomerativeClustering
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
cudnn.benchmark = True


def get_cos_similar_matrix(v1, v2):
    num = torch.mm(v1,torch.transpose(v2, 0, 1)) 
    denom = torch.linalg.norm(v1,  dim=1).reshape(-1, 1) * torch.linalg.norm(v2,  dim=1)
    res = num / denom
    res[torch.isnan(res)] = 0
    return res

def norm_matrix(act):
    row_mins=torch.min(act,1).values[:, None]
    row_maxs=torch.max(act,1).values[:, None] 
    act=(act-row_mins)/(row_maxs-row_mins)
    return act

#-----------------------Function to Reshape the Extracted Feature Maps----------------------------------
# Users may need to adjust the reshape_transform function for different ViT Models
# For instance, in DEiT, the first two tokens are [CLS] and [Dist], should the patch tokens start from the third element, thus we shall have:
# result = tensor[:, 2:, :].reshape(tensor.size(0),height, width, tensor.size(2))
def reshape_function_vit(tensor, height=14, width=14):
    if tensor.shape[0] == 1:
        height=14; width=14
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))
    elif tensor.shape[1] == 1:
        if tensor.shape[-1] == 768: # Quilt
            height=7; width=7
        else:   #CLIP
            height=16; width=16
        # [257, 1, 1024]
        tensor = tensor.transpose(0,1)
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))
        print(tensor.shape)
        # [1, 257, 1024]
        print("================")
    if tensor.shape[0] == 8:    # languagebind [8,257,1024]
        height=16; width=16
        # tensor = tensor.transpose(0,1)
        # tensor = tensor.unsqueeze(0)    # [1, 257,8,1024]
        
        tensor = tensor[0]
        tensor = tensor.unsqueeze(0)    # [1, 257,1024]
        
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))
        # print(result.shape)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


#--------------------------------Function to make the ViT-CX explanation-----------------------------
'''
1. model: ViT model to be explained;
2. image: input image in the tensor form (shape: [1,#channels,width,height]);
3. target_layer: the layer to extract feature maps  (e.g. model.blocks[-1].norm1);
4. target_category: int (class to be explained), in default - the top_1 prediction class;
5. distance_threshold: float between [0,1], distance threshold to make the clustering where  
   feature maps with similarity<distance_threshold will be merged together, in default - 0.1; 
6. reshape_function: function to reshape the extracted feature maps, in default - a reshape function for vanilla vit;
7. gpu_batch: batch size the run the prediction for the masked images, in default - 50.
'''

def ViT_CX(model,image,target_layer,target_category=None,distance_threshold=0.1,reshape_function=reshape_function_vit,gpu_batch=50):
    image=image.cuda()
    model_softmax=copy.deepcopy(model)
    model=model.eval()
    model=model.cuda()
    model_softmax = nn.Sequential(model_softmax, nn.Softmax(dim=1))
    model_softmax = model_softmax.eval()
    model_softmax = model_softmax.cuda()
    for p in model_softmax.parameters():
        p.requires_grad = False
    y_hat = model_softmax(image)
    y_hat_1=y_hat.detach().cpu().numpy()[0]
    if target_category==None:
        top_1=np.argsort(y_hat_1)[::-1][0]
        target_category = top_1
    class_p=y_hat_1[target_category]
    input_size=(image.shape[2],image.shape[3])
    transform_fp = transforms.Compose([transforms.Resize(input_size)])


    # Extract the ViT feature maps 
    GetFeatureMap= get_feature_map(model=model,target_layers=[target_layer],use_cuda=True,reshape_transform=reshape_function)
    _ = GetFeatureMap(input_tensor=image,target_category=int(target_category))
    feature_map=GetFeatureMap.featuremap_and_grads.featuremaps[0][0].cuda()

    # Reshape and normalize the ViT feature maps to get ViT masks
    feature_map=transform_fp(feature_map)
    mask=norm_matrix(torch.reshape(feature_map, (feature_map.shape[0],input_size[0]*input_size[1])))


    # Compute the pairwise cosine similarity and distance of the ViT masks
    similarity = get_cos_similar_matrix(mask,mask)
    distance = 1 - similarity

    # Apply the  AgglomerativeClustering with a given distance_threshold
    cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=distance_threshold,affinity='precomputed', linkage='complete') 
    cluster.fit(distance.cpu())
    cluster_num=len(set(cluster.labels_))
    print('number of masks after the clustering:'+str(cluster_num))

    # Use the sum of a clustering as a representation of the cluster
    cluster_labels=cluster.labels_
    cluster_labels_set=set(cluster_labels)
    mask_clustering=torch.zeros((len(cluster_labels_set),input_size[0]*input_size[1])).cuda()
    for i in range(len(mask)):
        mask_clustering[cluster_labels[i]]+=mask[i]

    # normalize the masks
    mask_clustering_norm=norm_matrix(mask_clustering).reshape((len(cluster_labels_set),input_size[0],input_size[1]))
    
    # compute the causal impact score
    compute_causal_score = causal_score(model_softmax, (input_size[0], input_size[1]),gpu_batch=gpu_batch)
    sal = compute_causal_score(image,mask_clustering_norm, class_p)[target_category].cpu().numpy()

    return sal