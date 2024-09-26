import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from xplique.wrappers import TorchWrapper
from xplique.metrics import MuFidelity

import clip

import torch
import torchvision.transforms as transforms

from tqdm import tqdm

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4048)]
)

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

def transform_vision_data(image_path, channel_first=False):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image =cv2.imread(image_path)
    
    image = Image.fromarray(image)
    image = data_transform(image)
    if channel_first:
        pass
    else:
        image = image.permute(1,2,0)
    return image.numpy()

def parse_args():
    parser = argparse.ArgumentParser(description='Deletion Metric')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/imagenet/ILSVRC2012_img_val',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/imagenet/val_clip_vitl_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        default=-1,
                        help='Datasets.')
    parser.add_argument('--explanation-smdl', 
                        type=str, 
                        default='./submodular_results/imagenet-clip-vitl-efficientv2/slico-0.0-0.05-1.0-1.0-pending-samples-8/npy',
                        # default='./submodular_results/celeba/random_patch-7x7-48/npy',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class CLIPModel_Super(torch.nn.Module):
    def __init__(self, 
                 type="ViT-L/14", 
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)
        
    def equip_semantic_modal(self, modal_list):
        text = clip.tokenize(modal_list).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
    def forward(self, vision_inputs):
        
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        scores = (image_features @ self.text_features.T).softmax(dim=-1)
        return scores.float()

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i
        
        # single_mask = cv2.resize(single_mask, (7,7))    # for smooth
        # single_mask = cv2.resize(single_mask, (224,224))
        # single_mask = np.exp(single_mask / single_mask.max() / 0.5)
        
        batch_mask.append(single_mask.astype(np.float32))

    return np.array(batch_mask).mean(-1)

def main(args):
    class_number = 1000
    batch = 2048
    
    # data preproccess
    with open(args.eval_list, "r") as f:
        datas = f.read().split('\n')

    label = []
    input_image = []
    smdl_mask = []
    for data in tqdm(datas[ : args.eval_number]):
        label.append(int(data.strip().split(" ")[-1]))
        input_image.append(
            transform_vision_data(os.path.join(args.Datasets, data.split(" ")[0]))
        )
        smdl_mask.append(
            np.load(
                os.path.join(os.path.join(args.explanation_smdl, data.strip().split(" ")[-1]), data.split(" ")[0].replace(".JPEG", ".npy")))    
        )
    label_onehot = tf.one_hot(np.array(label), class_number)
    input_image = np.array(input_image)
    smdl_mask = np.array(smdl_mask)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis_model = CLIPModel_Super("ViT-L/14", download_root=".checkpoints/CLIP")
    vis_model.eval()
    vis_model.to(device)
    
    semantic_path = "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)

    vis_model.text_features = semantic_feature
    
    model = TorchWrapper(vis_model.eval(), device)
    torch.cuda.empty_cache()

    # original
    metric = MuFidelity(model, input_image, label_onehot, batch_size=32, nb_samples=32, grid_size=7)
    
    batch_mask = convert_smdl_mask(smdl_mask)
    mufidelity_score = metric(batch_mask.astype(np.float32))
    print("Our Method MuFidelity Score: {}".format(mufidelity_score))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)