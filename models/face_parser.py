from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation.segmentation import deeplabv3_resnet50
import cv2

class FaceParser(nn.Module):
    def __init__(self, num_classes=9, model_path="./ckpt/FaceParser.ckpt"):
        super().__init__()
        self.mask_modes = ['mse', 'shape', 'blend', 'dynamic']
        # 定义模型
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
        self.model = load_lightning_dict(self.model, model_path)
        # 掩膜
        self.all_masks = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']
        # 索引
        self.mask2idx = {k: self.all_masks.index(k) for k in self.all_masks}

        self.downsample_cfg = {'size': (512, 512), 'mode': 'area'}
        self.upsample_cfg = {'size': (1024, 1024), 'mode': 'bilinear'}
        self.mask_idxs = {'mse': [self.mask2idx['skin']],
                          'blend': [self.mask2idx['skin']],
                          'shape': [],
                          'dynamic': []}

    def set_idx_list(self, attributes):
        for attr in attributes:
            self.set_idx(attr)

    def set_idx(self, attribute):
        """
        貌似没啥用，给定属性对应位置
        """
        if attribute in ['wearing_lipstick', 'mouth_slightly_open', 'smiling', 'big_lips']:
            target_name = 'mouth'
        elif attribute in ['bushy_eyebrows', 'arched_eyebrows']:
            target_name = 'eyebrows'
        elif attribute in ['narrow_eyes']:
            target_name = 'eyes'
        elif attribute in ['pointy_nose', 'big_nose']:
            target_name = 'nose'
        elif attribute in ['black_hair', 'brown_hair', 'blond_hair', 'gray_hair', 'wavy_hair', 'straight_hair']:
            target_name = 'hair'
        else:
            raise ValueError('attribute not found')

        target_idx = self.mask2idx[target_name]
        self.mask_idxs['blend'] += [target_idx]
        self.mask_idxs['shape'] += [target_idx]
        self.mask_idxs['dynamic'] += [target_idx]

        if attribute in ('straight_hair', 'wavy_hair'):  # hair shape requires careful treatment, especially with ear component
            self.mask_idxs['mse'] += [self.mask2idx['ears']]
            self.mask_idxs['blend'] += [self.mask2idx['ears']]
            self.mask_idxs['shape'] += [self.mask2idx['ears']]

    def forward(self, img):
        if img.size(-1) != 512:
            img = F.interpolate(img, **self.downsample_cfg)
        img = img * 2 - 1
        parsed_face = F.softmax(self.model(img)['out'], dim=1)

        return parsed_face

def load_lightning_dict(model, ckpt_file):
    # thisdir = Path(__file__).parent
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'model.' in k:
            new_state_dict[k[6:]] = v
        elif 'loss.weight' in k:
            continue
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.eval().requires_grad_(False)
    return model

def interpolate(img, size):
    if type(size) == tuple:
        assert size[0] == size[1]
        size = size[0]

    orig_size = img.size(3)
    if size < orig_size:
        mode = 'area'
    else:
        mode = 'bilinear'
    return F.interpolate(img, (size, size), mode=mode)

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 1024:
        img = interpolate(img, 1024)
    return img

def preproccess(image):
    img = Image.fromarray(image)
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 1024:
        img = interpolate(img, 1024)
    return img

def remove(image_path, model):
    input = read_img(image_path)
    parsed_face = model(input)
    mask = parsed_face.cpu().numpy()[0]# * 255

    mask_ = 1 - mask

    return mask_