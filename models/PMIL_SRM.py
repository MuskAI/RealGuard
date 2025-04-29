import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .srm_filter_kernel import all_normalized_hpf_list
class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   

    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)

    return output





# -------------------- Patch Splitter --------------------
def split_into_patches(image, patch_size=256):
    """
    image: Tensor [C, H, W]
    return: patches: [N, C, patch_size, patch_size], positions: List[(top, left)]
    """
    C, H, W = image.shape
    stride = patch_size

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    padded = F.pad(image, (0, pad_w, 0, pad_h))  # pad right and bottom

    _, H_pad, W_pad = padded.shape
    patches = padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride)  # [C, nH, nW, patch, patch]
    nH, nW = patches.shape[1], patches.shape[2]

    patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # [nH, nW, C, patch, patch]
    patches = patches.view(-1, C, patch_size, patch_size)  # [N_patches, C, patch, patch]

    # 计算每个 patch 左上角坐标
    positions = [(i * stride, j * stride) for i in range(nH) for j in range(nW)]

    return patches, positions

# -------------------- ResNet Components --------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# -------------------- PMIL Model --------------------
class PMIL_Model(nn.Module):
    def __init__(self, resnet_path=None):
        super(PMIL_Model, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
        self.hpf = HPF()
        self.load_resnet_weights(resnet_path)
        # 第一种方法
        # if resnet_path is not None:
        #     pretrained_dict = torch.load(resnet_path, map_location='cpu')
        #     model_dict = self.model.state_dict()
        #     for k in pretrained_dict:
        #         if k in model_dict and pretrained_dict[k].size() == model_dict[k].size():
        #             model_dict[k] = pretrained_dict[k]
        #     self.model.load_state_dict(model_dict)
        #     print("Loaded pretrained model from %s" % resnet_path)
        # 第二种方法
        # if resnet_path is not None:
        #     print(f"Loading checkpoint from {resnet_path} ...")
        #     checkpoint = torch.load(resnet_path, map_location='cpu')

        #     # 取出模型参数部分
        #     state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        #     # 去掉 'model.' 前缀
        #     new_state_dict = {}
        #     for k, v in state_dict.items():
        #         if k.startswith('model.'):
        #             new_k = k[len('model.'):]  # 去掉前缀
        #         else:
        #             new_k = k
        #         new_state_dict[new_k] = v

        #     # 加载到当前模型
        #     msg = self.model.load_state_dict(new_state_dict, strict=False)
        #     print("✅ Model loaded.")
        #     print("🔍 Missing keys:", msg.missing_keys)
        #     print("🔍 Unexpected keys:", msg.unexpected_keys)
    def load_resnet_weights(self, resnet_path):
        if resnet_path is None:
            return

        print(f"🔄 Trying to load pretrained weights from {resnet_path} ...")

        pretrained_dict = torch.load(resnet_path, map_location='cpu')
        model_dict = self.model.state_dict()

        # 第一种方式：只加载尺寸匹配的
        matched, total = 0, 0
        for k in pretrained_dict:
            if k in model_dict:
                total += 1
                if pretrained_dict[k].size() == model_dict[k].size():
                    model_dict[k] = pretrained_dict[k]
                    matched += 1

        match_ratio = matched / total if total > 0 else 0

        if match_ratio > 0.7:  # 超过70%匹配，使用第一种方式
            self.model.load_state_dict(model_dict)
            print(f"✅ Loaded pretrained model with matched ratio {match_ratio:.2%} from {resnet_path}")
        else:
            print(f"⚠️ Matched ratio too low ({match_ratio:.2%}), switching to second loading strategy ...")

            # 第二种方式：去掉前缀，并使用 strict=False
            checkpoint = pretrained_dict
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']

            new_state_dict = {}
            for k, v in checkpoint.items():
                new_k = k[len('model.'):] if k.startswith('model.') else k
                new_state_dict[new_k] = v

            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print("✅ Model loaded with fallback strategy.")
            print("🔍 Missing keys:", msg.missing_keys)
            print("🔍 Unexpected keys:", msg.unexpected_keys)
    def forward(self, images):
        srm_images = self.hpf(images)
        output = self.model(srm_images)
        return output

# -------------------- Entry --------------------
def PMIL_SRM(resnet_path=None):
    return PMIL_Model(resnet_path)