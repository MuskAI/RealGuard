import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

Perturbations = K.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.3),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.3),
    K.RandomGamma(gamma=(0.7, 1.5), p=0.3),
    K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.3),
    K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
    K.RandomMotionBlur(kernel_size=5, angle=15.0, direction=0.5, p=0.2),
    # K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.2),
    K.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), p=0.3),
)
transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    # transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_mil_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

transform_mil_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
# added by haoran
transform_mil_train_fix = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.5)),  # 根据实际情况调整 scale
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
class ResizeIfSmall:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img: Image.Image):
        w, h = img.size
        if min(h, w) < self.min_size:
            scale = self.min_size / min(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

transform_mil_test_fix = transforms.Compose([
    ResizeIfSmall(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#############################################################################

################## Setting 1 #####################
# 这里似乎并没有100%复现CNNSpot的功能
transform_patch_based_train = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.5)),  # 根据实际情况调整 scale
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
transform_patch_based_test = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

########### 100%复刻CNNSpot的处理方式 ###########
transform_cnnspot_train = transforms.Compose([
    transforms.Resize(256), # 这是关键
    transforms.RandomCrop(224), # 这是关键
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
transform_cnnspot_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)
transform_cnnspot_test_noresize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)
##############################################

####### anyres #########

def anyres_rz(img):
    """
    anyres模块
    1.如果输入图片的分辨率H,W都小于256 则不分patch，直接处理
    2.如果输入图片的分辨率H,W有一个小于256 则保持分辨率进行reszie
    3.如果输入图片的分辨率H,W都大于256 则进行筛分patch
    4.如果输入图片的分辨率H,W都大于2048 则保持分辨率进行reszie 最高限制2k*2k
    Args:
        img (PIL Image): Image to be scaled.
    Returns:
        PIL Image: Rescaled image.
    """
    w, h = img.size
    if w < 256 and h < 256:
        scale = 256 / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)
    elif w < 256 or h < 256:
        scale = 256 / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)
    elif w >= 2048 or h >= 2048:
        scale = 2048 / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)
    else:
        return img

transform_mil_anyres_test = transforms.Compose([
    transforms.Lambda(anyres_rz),  # 自适应尺寸处理
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # HACK 
    ]
)
#######################

# from xiaohongshu KDD2025
class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)
###################


#### More powerful argument
def resize_or_crop(img):
    w, h = img.size
    if w > 512 and h > 512:
        return transforms.RandomCrop((512, 512))(img)
    else:
        return transforms.Resize((512, 512))(img)

# def resize_or_crop(img, 
#                    resize_to=(256, 256), 
#                    scale_range=(0.5, 1.5), 
#                    scale_prob=0.5, 
#                    upsample_methods=None):
#     """
#     根据图像大小决定是否resize或缩放图像。

#     参数：
#     - img: 输入图像（PIL Image）
#     - resize_to: 小图像resize的目标大小
#     - scale_range: 大图像缩放的比例范围（min, max），默认是[0.5, 1.5]
#     - scale_prob: 大图像执行缩放的概率
#     - upsample_methods: 用于上采样的小图的插值方法列表
#     """
#     w, h = img.size
#     if w < 256 and h < 256:
#         # 对小图进行上采样
#         if upsample_methods is None:
#             upsample_methods = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.NEAREST]
#         method = random.choice(upsample_methods)
#         return img.resize(resize_to, method)

#     elif w > 256 and h > 256:
#         # 对大图，有概率进行缩放
#         if random.random() < scale_prob:
#             scale_factor = random.uniform(*scale_range)
#             new_w = int(w * scale_factor)
#             new_h = int(h * scale_factor)
#             upsample_methods = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.NEAREST]
#             method = random.choice(upsample_methods)
#             return img.resize((new_w, new_h), method)
#         else:
#             return img  # 保持原始分辨率不变
#     else:
#         # 一边小一边大时，统一resize到指定大小
#         return img.resize(resize_to, Image.BILINEAR)
transform_mil_train_plus = transforms.Compose([
    transforms.Lambda(resize_or_crop),  # 自适应尺寸处理
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),  # 保留你的自定义扰动
    RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)




