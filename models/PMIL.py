import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
            model_dict = self.model.state_dict()
            for k in pretrained_dict:
                if k in model_dict and pretrained_dict[k].size() == model_dict[k].size():
                    model_dict[k] = pretrained_dict[k]
            self.model.load_state_dict(model_dict)

    def forward(self, images):
        """
        images: Tensor of shape [B, 3, H, W]
        returns: List of dicts with keys:
            'scores': Tensor [N_patches, 2],
            'positions': List[(top, left)]
        """
        
        # delete this in the future
        # B, C, H, W = images.shape
        # results = []

        # # 这不是脱裤子放屁吗？ 之后会删掉的
        # for i in range(B):
        #     patches, positions = split_into_patches(images[i])  # [N_patches, C, 256, 256], List of (top, left)
        #     patches = patches.to(images.device)
        #     scores = self.model(patches)              # [N_patches, 2]
        #     # results.append({"scores": scores, "positions": positions})
        #     results.append(scores)
        
        # output = torch.cat(results, dim=0)  # 把 list 拼接成一个 tensor
        
        
        output = self.model(images)
        
        return output

# -------------------- Entry --------------------
def PMIL(resnet_path=None):
    return PMIL_Model(resnet_path)