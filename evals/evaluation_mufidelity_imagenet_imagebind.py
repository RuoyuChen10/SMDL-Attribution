import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from xplique.wrappers import TorchWrapper
from xplique.metrics import MuFidelity

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

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
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
                        default='datasets/imagenet/val_imagebind_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        default=1000,
                        help='Datasets.')
    parser.add_argument('--explanation-method', 
                        type=str, 
                        default='./explanation_results/imagenet-imagebind-true/Rise',
                        help='Save path for saliency maps generated by interpretability methods.')
    parser.add_argument('--explanation-smdl', 
                        type=str, 
                        default='./submodular_results/imagenet-imagebind/slico-0.0-1.0-1.0-10.0/npy',
                        # default='./submodular_results/celeba/random_patch-7x7-48/npy',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
    def mode_selection(self, mode):
        if mode not in ["text", "audio", "thermal", "depth", "imu"]:
            print("mode {} does not comply with the specification, please select from \"text\", \"audio\", \"thermal\", \"depth\", \"imu\".".format(mode))
        else:
            self.mode = mode
            print("Select mode {}".format(mode))
            
    def equip_semantic_modal(self, modal_list):
        if self.mode == "text":
            self.semantic_modal = data.load_and_transform_text(modal_list, self.device)
        elif self.mode == "audio":
            self.semantic_modal = data.load_and_transform_audio_data(modal_list, self.device)
        
        input = {
                # "vision": vision_inputs,
                self.mode: self.semantic_modal
            }
        with torch.no_grad():
            self.semantic_modal = self.base_model(input)[self.mode]
        print("Equip with {} modal.".format(self.mode))
        
    def forward(self, vision_inputs):
        inputs = {
            "vision": vision_inputs,
        }
        
        with torch.no_grad():
            embeddings = self.base_model(inputs)
        
        scores = torch.softmax(embeddings["vision"] @ self.semantic_modal.T, dim=-1)
        return scores

def zeroshot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = data.load_and_transform_text(texts, device) #tokenize
            input = {
                "text": texts
            }
            with torch.no_grad():
                class_embeddings = model(input)["text"]

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights).cuda()
    return zeroshot_weights  

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i
        
        single_mask = cv2.resize(single_mask, (7,7))    # for smooth
        single_mask = cv2.resize(single_mask, (224,224))
        # single_mask = np.exp(single_mask / single_mask.max() / 0.02)
        
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
    explanations = []
    smdl_mask = []
    for data in tqdm(datas[ : args.eval_number]):
        label.append(int(data.strip().split(" ")[-1]))
        input_image.append(
            transform_vision_data(os.path.join(args.Datasets, data.split(" ")[0]))
        )
        explanations.append(
            np.load(
                os.path.join(args.explanation_method, data.split(" ")[0].replace(".JPEG", ".npy")))
        )
        smdl_mask.append(
            np.load(
                os.path.join(os.path.join(args.explanation_smdl, data.strip().split(" ")[-1]), data.split(" ")[0].replace(".JPEG", ".npy")))    
        )
    label_onehot = tf.one_hot(np.array(label), class_number)
    input_image = np.array(input_image)
    explanations = np.array(explanations)
    smdl_mask = np.array(smdl_mask)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = imagebind_model.imagebind_huge(pretrained=True)
    torch_model.eval()
    torch_model.to(device)
    
    vis_model = ImageBindModel_Super(torch_model, device)
    vis_model.mode_selection("text")
    
    semantic_path = "ckpt/semantic_features/imagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
        vis_model.semantic_modal = semantic_feature
    
    model = TorchWrapper(vis_model.eval(), device)
    
    torch.cuda.empty_cache()

    # original
    metric = MuFidelity(model, input_image, label_onehot, batch_size=32, nb_samples=32, grid_size=10)

    mufidelity_score_org = metric(explanations)
    
    batch_mask = convert_smdl_mask(smdl_mask)
    mufidelity_score = metric(batch_mask)
    print("Original Attribution Method MuFidelity Score: {}".format(mufidelity_score_org))
    print("Our Method MuFidelity Score: {}".format(mufidelity_score))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)