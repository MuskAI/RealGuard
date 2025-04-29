# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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
from .patch_splitter import PatchifyImage

from torchvision.transforms.functional import to_tensor

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


####### anyres #########

def anyres_rz(img):
    """
    anyres模块
    1. 如果输入图片的分辨率H,W都大于1024，则等比resize到1024
    2. 如果输入图片的分辨率H,W都小于256 则不分patch，直接处理
    3. 如果输入图片的分辨率H,W有一个小于256 则保持分辨率进行resize
    4. 如果输入图片的分辨率H,W都大于256 则进行筛分patch

    Args:
        img (PIL Image): Image to be scaled.

    Returns:
        PIL Image: Rescaled image.
    """
    w, h = img.size

    # 新增：如果H和W都大于1024，等比例缩放到1024
    if w > 1024 and h > 1024:
        scale = 1024 / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size  # 更新尺寸信息以用于后续判断

    if w < 256 and h < 256:
        return img
    if w < 256 or h < 256:
        scale = 256 / min(w, h)
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

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path
        self.patchify = PatchifyImage(patch_size=256)  # 你上面定义好的类
        self.data_list = []

        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
    
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake'] or os.listdir(os.path.join(file_path, folder_name)) == ['1_fake', '0_real']
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, target = sample['image_path'], sample['label']

        # 打开图像
        image = Image.open(image_path).convert('RGB')

        # === Step 1: Anyres 自适应处理 ===
        image = anyres_rz(image)  # PIL.Image

        # === Step 2: 转为 tensor，进行 patchify ===
        image_tensor = to_tensor(image)  # [C, H, W]
        patches, coords, pad, stride = self.patchify(image_tensor)  # [N, C, 256, 256]

        # === Step 3: 对每个 patch 做 DCT + transform ===
        all_views = []
        for patch_tensor in patches:
           
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(patch_tensor)
            x_0 = transform_train(patch_tensor)

            x_minmin = transform_train(x_minmin)
            x_maxmax = transform_train(x_maxmax)
            x_minmin1 = transform_train(x_minmin1)
            x_maxmax1 = transform_train(x_maxmax1)

            views = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)  # [5, C, H, W]
            all_views.append(views)

        all_views = torch.stack(all_views, dim=0)  # [N_patches, 5, C, 256, 256]
        # 将target 复制 N_patches 次
        target = torch.tensor(int(target)).repeat(len(all_views))
        return all_views, target