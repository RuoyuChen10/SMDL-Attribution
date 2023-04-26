import torch.nn as nn
import math
import torch
import torch.nn.functional as F

from collections import OrderedDict

__all__ = ['ResNet']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, 2)
        self.fc2 = nn.Linear(512 * block.expansion, 3)
        self.fc3 = nn.Linear(512 * block.expansion, 3)
        self.fc4 = nn.Linear(512 * block.expansion, 3)
        self.fc5 = nn.Linear(512 * block.expansion, 3)
        self.fc6 = nn.Linear(512 * block.expansion, 3)
        self.fc7 = nn.Linear(512 * block.expansion, 3)
        self.fc8 = nn.Linear(512 * block.expansion, 3)
        self.fc9 = nn.Linear(512 * block.expansion, 5)
        self.fc10 = nn.Linear(512 * block.expansion, 2)
        self.fc11 = nn.Linear(512 * block.expansion, 2)
        self.fc12 = nn.Linear(512 * block.expansion, 2)
        self.fc13 = nn.Linear(512 * block.expansion, 2)
        self.fc14 = nn.Linear(512 * block.expansion, 3)
        self.fc15 = nn.Linear(512 * block.expansion, 3)
        self.fc16 = nn.Linear(512 * block.expansion, 3)
        self.fc17 = nn.Linear(512 * block.expansion, 3)
        self.fc18 = nn.Linear(512 * block.expansion, 3)
        self.fc19 = nn.Linear(512 * block.expansion, 3)
        self.fc20 = nn.Linear(512 * block.expansion, 3)
        self.fc21 = nn.Linear(512 * block.expansion, 3)
        self.fc22 = nn.Linear(512 * block.expansion, 3)
        self.fc23 = nn.Linear(512 * block.expansion, 3)
        self.fc24 = nn.Linear(512 * block.expansion, 3)
        self.fc25 = nn.Linear(512 * block.expansion, 3)
        self.fc26 = nn.Linear(512 * block.expansion, 3)
        self.fc27 = nn.Linear(512 * block.expansion, 3)
        self.fc28 = nn.Linear(512 * block.expansion, 3)
        self.fc29 = nn.Linear(512 * block.expansion, 3)
        self.fc30 = nn.Linear(512 * block.expansion, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        x7 = self.fc7(x)
        x8 = self.fc8(x)
        x9 = self.fc9(x)
        x10 = self.fc10(x)
        x11 = self.fc11(x)
        x12 = self.fc12(x)
        x13 = self.fc13(x)
        x14 = self.fc14(x)
        x15 = self.fc15(x)
        x16 = self.fc16(x)
        x17 = self.fc17(x)
        x18 = self.fc18(x)
        x19 = self.fc19(x)
        x20 = self.fc20(x)
        x21 = self.fc21(x)
        x22 = self.fc22(x)
        x23 = self.fc23(x)
        x24 = self.fc24(x)
        x25 = self.fc25(x)
        x26 = self.fc26(x)
        x27 = self.fc27(x)
        x28 = self.fc28(x)
        x29 = self.fc29(x)
        x30 = self.fc30(x)
        return [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30]

class AttributeNet(nn.Module):
    """
    AttributeNet
    @Author: Ruoyu Chen
    """
    def __init__(self, 
        pretrained = "/home/lsf/桌面/MaskFaceGAN/models/vgg_attribute_classifier/AttributeNet.pkl"):
        super().__init__()
        self.img_size = 224
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.pretrained = pretrained
        self.init_model()
        self.model.eval()   # 冻结参数
        self.Face_attributes_name = [
            "Gender","Age","Race","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]
        self.Gender = ["Male","Female"]
        self.Age = ["Young","Middle Aged","Senior"]
        self.Race = ["Asian","White","Black"]
        self.Hair_color = ["Black Hair","Blond Hair","Brown Hair","Gray Hair","Unknown Hair"]

        self.desired_attribute = [
            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]# currently Sideburns not support
        self.softmax = nn.Softmax(dim=1)

    def init_model(self):
        """
        Init model params
        """
        # No related
        model_dict = self.model.state_dict()
        pretrained_param = torch.load(self.pretrained, map_location=torch.device('cpu'))
        try:
            pretrained_param = pretrained_param.state_dict()
        except:
            pass

        new_state_dict = OrderedDict()
        for k, v in pretrained_param.items():
            if k in model_dict:
                new_state_dict[k] = v
                print("Load parameter {}".format(k))
            elif k[7:] in model_dict:
                new_state_dict[k[7:]] = v
                print("Load parameter {}".format(k[7:]))

        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)
        print("Success load pre-trained face model {}".format(self.pretrained))

    def set_idx_list(self, attribute=["Male", "Young", "Middle Aged", "Senior", "Asian","White","Black"]):
        self.desired_attribute = attribute

    def set_idx(self, attr):
        self.desired_attribute += [attr]

    def forward(self, x):
        if x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear')
        output = self.model(x)

        out = []

        for attribute in self.desired_attribute:
            # select attribute
            # Gender
            if attribute in self.Gender:
                out.append(
                    self.softmax(output[0])[:,self.Gender.index(attribute)].unsqueeze(1)
                    )
            elif attribute in self.Age:
                out.append(
                    self.softmax(output[1])[:,self.Age.index(attribute)].unsqueeze(1)
                    )
            elif attribute in self.Race:
                out.append(
                    self.softmax(output[2])[:,self.Race.index(attribute)].unsqueeze(1)
                    )
            elif attribute in self.Hair_color:
                out.append(
                    self.softmax(output[8])[:,self.Hair_color.index(attribute)].unsqueeze(1)
                    )
            else:
                out.append(
                    self.softmax(output[self.Face_attributes_name.index(attribute)])[:,0].unsqueeze(1)
                    )

        out = torch.cat(out, 1)

        return out

# model = AttributeNet()
# x = torch.randn([8,3,224,224])
# model.set_desired_attribute()
# output = model(x)
# print(output.shape)