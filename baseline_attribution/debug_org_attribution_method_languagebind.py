import os
import cv2
import math
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

from tqdm import tqdm
import json
from utils import *

results_save_root = "./explanation_insertion_results"
explanation_method = "explanation_results/imagenet-languagebind-true/ViT-CX"
image_root_path = "datasets/imagenet/ILSVRC2012_img_val"
eval_list = "datasets/imagenet/val_languagebind_5k_true.txt"
save_doc = "imagenet-true-languagebind"
steps = 50
batch_size = 10
image_size_ = 224

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
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
            
        scores = torch.softmax(embeddings["video"] @ self.semantic_modal.T, dim=-1)
        return scores

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

def preprocess_input(images):
    """
    Input:
        image: An image read by opencv [b,w,h,c]
    Output:
        outputs: After preproccessing, is a tensor [c,w,h]
    """
    outputs = []
    for image in images:
        image = Image.fromarray(image)
        image = data_transform(image)
        outputs.append(image)
    return torch.stack(outputs)

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    mkdir(results_save_root)
    save_dir = os.path.join(results_save_root, save_doc)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, explanation_method.split("/")[-1])
    mkdir(save_dir)

   # Load model
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'thermal': 'LanguageBind_Thermal',
        'image': 'LanguageBind_Image',
        'depth': 'LanguageBind_Depth',
    }
    model_bind = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    model_bind = model_bind.to(device)
    model_bind.eval()

    semantic_path = "ckpt/semantic_features/languagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
        
    model = LanguageBindModel_Super(model_bind, device)
    print("load languagebind model")

    model.semantic_modal = semantic_feature

    with open(eval_list, "r") as f:
        infos = f.read().split('\n')

    for info in tqdm(infos):
        json_file = {}
        class_index = int(info.split(" ")[-1])
        
        if "ILSVRC2012_val_00020699" not in info:
            continue
        
        # if class_index!=351:
        #     continue
        
        image_path = os.path.join(image_root_path, info.split(" ")[0])

        mask_path = os.path.join(explanation_method, info.split(" ")[0].replace(".JPEG", ".npy"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size_, image_size_))
        explanation = np.load(mask_path)

        insertion_explanation_images = []
        deletion_explanation_images = []
        for i in range(1, steps+1):
            perturbed_rate = i / steps
            insertion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "insertion"))
            deletion_explanation_images.append(perturbed(image, explanation, rate = perturbed_rate, mode = "deletion"))
        
        insertion_explanation_images_input = preprocess_input(
            np.array(insertion_explanation_images)
        ).to(device)
        deletion_explanation_images_input = preprocess_input(
            np.array(deletion_explanation_images)
        ).to(device)

        batch_step = math.ceil(
            insertion_explanation_images_input.shape[0] / batch_size)
        
        insertion_data = []
        deletion_data = []
        for j in range(batch_step):
            insertion_explanation_images_input_results = model(
                insertion_explanation_images_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            insertion_data += insertion_explanation_images_input_results.cpu().numpy().tolist()
            
            deletion_explanation_images_input_results = model(
                deletion_explanation_images_input[j*batch_size:j*batch_size+batch_size])[:,class_index]
            deletion_data += deletion_explanation_images_input_results.cpu().numpy().tolist()
        
        json_file["consistency_score"] = insertion_data
        json_file["collaboration_score"] = deletion_data
        
        save_path = os.path.join(
            save_dir, info.split(" ")[0].replace(".JPEG", ".json")
        )
        with open(save_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        
    return

main()