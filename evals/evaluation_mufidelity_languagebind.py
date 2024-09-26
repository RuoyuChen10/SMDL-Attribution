import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from xplique.wrappers import TorchWrapper
from xplique.metrics import MuFidelity

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

import torch
import torchvision.transforms as transforms

from tqdm import tqdm

# tf.config.run_functions_eagerly(True)

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4048)]
# )

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
                        default='datasets/imagenet/val_languagebind_5k_true.txt',
                        help='Datasets.')
    parser.add_argument('--eval-number',
                        type=int,
                        default=1000,
                        help='Datasets.')
    parser.add_argument('--explanation-smdl', 
                        type=str, 
                        default='./submodular_results/imagenet-languagebind/slico-0.0-0.05-1.0-1.0/npy',
                        # default='./submodular_results/celeba/random_patch-7x7-48/npy',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class LanguageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model, device,
                 pretrained_ckpt = f'lb203/LanguageBind_Image',):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            pretrained_ckpt, cache_dir='.checkpoints/tokenizer_cache_dir')
        
        self.clip_type = ["video"]
        self.modality_transform = {c: transform_dict[c](self.base_model.modality_config[c]) for c in self.clip_type}
    
    # def mode_selection(self, mode):
    #     if mode not in ["image", "audio", "video", "depth", "thermal", "language"]:
    #         print("mode {} does not comply with the specification, please select from \"image\", \"audio\", \"video\", \"depth\", \"thermal\", \"language\".".format(mode))
    #     else:
    #         self.mode = mode
    #         print("Select mode {}".format(mode))
    
    # def equip_semantic_modal(self, modal_list):
    #     if self.mode == "language":
    #         self.semantic_modal = to_device(self.tokenizer(modal_list, max_length=77, padding='max_length',
    #                                          truncation=True, return_tensors='pt'), self.device)
    #     elif self.mode in self.clip_type:
    #         self.semantic_modal = to_device(self.modality_transform[self.mode](modal_list), self.device)
        
    #     input = {
    #             # "vision": vision_inputs,
    #             self.mode: self.semantic_modal
    #         }
    #     with torch.no_grad():
    #         self.semantic_modal = self.base_model(input)[self.mode]
    #     print("Equip with {} modal.".format(self.mode))
    
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

def convert_smdl_mask(smdl_mask):
    batch_mask = []
    for smdl_single_mask in smdl_mask:
        single_mask = np.zeros_like(smdl_single_mask[0])
        length = smdl_single_mask.shape[0]
        for i in range(length):
            single_mask[smdl_single_mask[i]>0] = length - i
        
        # single_mask = cv2.resize(single_mask, (7,7))    # for smooth
        # single_mask = cv2.resize(single_mask, (224,224))
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
    
    # Model Init
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load model
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        # 'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        # 'thermal': 'LanguageBind_Thermal',
        # 'image': 'LanguageBind_Image',
        # 'depth': 'LanguageBind_Depth',
    }
    torch_model = LanguageBind(clip_type=clip_type, cache_dir='.checkpoints')
    torch_model.eval()
    torch_model.to(device)

    vis_model = LanguageBindModel_Super(torch_model, device)
    print("load languagebind model")
    
    semantic_path = "ckpt/semantic_features/languagebind_imagenet_zeroweights.pt"
    if os.path.exists(semantic_path):
        semantic_feature = torch.load(semantic_path, map_location="cpu")
        semantic_feature = semantic_feature.to(device)
        vis_model.semantic_modal = semantic_feature

    model = TorchWrapper(vis_model.eval(), device)
    
    torch.cuda.empty_cache()

    # original
    metric = MuFidelity(model, input_image, label_onehot, batch_size=32, nb_samples=32, grid_size=7)

    batch_mask = convert_smdl_mask(smdl_mask)
    mufidelity_score = metric(batch_mask)
    
    print("Our Method on LanguageBind MuFidelity Score: {}".format(mufidelity_score))
    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)