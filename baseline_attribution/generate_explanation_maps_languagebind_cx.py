# -*- coding: utf-8 -*-  

"""
Created on 2024/4/15

@author: Ruoyu Chen
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

from .ViT_CX.ViT_CX import ViT_CX

import torch
from torchvision import transforms

from utils import *

# tf.config.run_functions_eagerly(True)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
# )

SAVE_PATH = "explanation_results/"
mkdir(SAVE_PATH)

mode = "imagenet"
net_mode  = "languagebind" # "resnet", vgg

if mode == "imagenet":
    if net_mode == "languagebind":
        img_size = 224
        dataset_index = "datasets/imagenet/val_languagebind_5k_true.txt"
        SAVE_PATH = os.path.join(SAVE_PATH, "imagenet-languagebind-true")
    # elif net_mode == "languagebind":
        
    dataset_path = "datasets/imagenet/ILSVRC2012_img_val"
    class_number = 1000
    batch = 100
    mkdir(SAVE_PATH)

data_transform = transforms.Compose(
        [
            transforms.Resize(
                (224,224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device,
                 pretrained_ckpt = f'lb203/LanguageBind_Image',):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')
        
        self.clip_type = ["video", "audio", "thermal", "image", "depth"]
        self.modality_transform = {c: transform_dict[c](self.base_model.modality_config[c]) for c in self.clip_type}
    
    def mode_selection(self, mode):
        if mode not in ["image", "audio", "video", "depth", "thermal", "language"]:
            print("mode {} does not comply with the specification, please select from \"image\", \"audio\", \"video\", \"depth\", \"thermal\", \"language\".".format(mode))
        else:
            self.mode = mode
            print("Select mode {}".format(mode))
    
    def equip_semantic_modal(self, modal_list):
        if self.mode == "language":
            self.semantic_modal = to_device(self.tokenizer(modal_list, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), self.device)
        elif self.mode in self.clip_type:
            self.semantic_modal = to_device(self.modality_transform[self.mode](modal_list), self.device)
        
        input = {
                # "vision": vision_inputs,
                self.mode: self.semantic_modal
            }
        with torch.no_grad():
            self.semantic_modal = self.base_model(input)[self.mode]
        print("Equip with {} modal.".format(self.mode))
    
    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: 
        """
        vision_inputs = vision_inputs.unsqueeze(2)
        vision_inputs = vision_inputs.repeat(1,1,8,1,1)
        inputs = {
            "video": {'pixel_values': vision_inputs},
        }
        
        # with torch.no_grad():
        embeddings = self.base_model(inputs)
            
        scores = torch.softmax(embeddings["video"] @ self.semantic_modal.T, dim=-1)
        return scores

def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_outputs = []
    
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        # image = image.unsqueeze(1).repeat(1, 8, 1, 1)
        image_outputs.append(image)
    image_outputs = torch.stack(image_outputs, dim=0)
    image_outputs = image_outputs.permute(0,2,3,1)
    return image_outputs.cpu().numpy()

def preprocess_image(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image = data_transform(image)
    return image.unsqueeze(0)

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = to_device(tokenizer(texts, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device) #tokenize
            input = {
                "language": texts
            }
            with torch.no_grad():
                class_embeddings = model(input)["language"]

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).cuda()
    return zeroshot_weights


def main():
    # Model Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # Load model
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'thermal': 'LanguageBind_Thermal',
        'image': 'LanguageBind_Image',
        'depth': 'LanguageBind_Depth',
    }
    model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    model = model.to(device)
    model.eval()
    
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')

    
    semantic_path = "ckpt/semantic_features/languagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    else:
        semantic_feature = zeroshot_classifier(model, imagenet_classes, imagenet_templates, tokenizer, device)
        torch.save(semantic_feature, semantic_path)
        
    vis_model = LanguageBindModel_Super(model, device)
    print("load languagebind model")
        
    vis_model.semantic_modal = semantic_feature
    
    # data preproccess
    with open(dataset_index, "r") as f:
        datas = f.read().split('\n')
    
    input_data = []
    label = []
    for data in datas:
        label.append(int(data.strip().split(" ")[-1]))
        input_data.append(
            os.path.join(dataset_path, data.split(" ")[0])
        )
    
    # explanation methods    
    explainer_method_name = "ViT-CX"
    exp_save_path = os.path.join(SAVE_PATH, explainer_method_name)
    mkdir(exp_save_path)
    
    target_layer=vis_model.base_model.modality_encoder.video.encoder.layers[-1].layer_norm1
    
    total_steps = math.ceil(len(input_data) / batch)
    
    for step in tqdm(range(len(input_data))):
        img_path = input_data[step]
        category = label[step]
        
        if os.path.exists(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".JPEG", ".npy"))):
            continue
        
        image = cv2.imread(img_path)
        image_cpu = preprocess_image(image)
        
        # Perform ViT-CX
        # try:
        result=ViT_CX(vis_model, 
                    image_cpu, 
                    target_layer, 
                    target_category=category, 
                    distance_threshold=0.1, 
                    gpu_batch=20)
        
        np.save(os.path.join(exp_save_path, img_path.split("/")[-1].replace(".JPEG", "")), result)
        # except:
        #     print("CUDA out of memory, skip!")
        #     torch.cuda.empty_cache()
    return

main()