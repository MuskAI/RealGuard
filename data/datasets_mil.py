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

# added by haoran
from tqdm import tqdm
#################

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
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
    transforms.Resize([512, 512]), # 暂时粗暴resize
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

transform_mil_test = transforms.Compose([
    transforms.Resize([512, 512]), # 暂时粗暴resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

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
def loadpathslist_genimage(root, flag, data_split="train",select_data_list=None):
    """
    @author: haoran
    @time: 2024/3/13 20:36
    @description: 从GenImage读取图片生成list
    """

    def get_all_image_paths(path, flag, data_split="train",select_data_list=None):
        """递归获取指定路径下的所有图片文件路径"""
        extensions = {".jpg", ".png", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"}

        subtasks = os.listdir(path)  # 获取path目录下的所有文件和文件夹
        # 如果subtasks中的内容在select_data_list里则选取，否则删除
        if select_data_list is not None:    
            subtasks = [subtask for subtask in subtasks if subtask in select_data_list]
        else:
            pass
        subtasks_paths = [os.path.join(path, task) for task in subtasks]  # 生成完整路径
        all_images_paths = []  # 存储所有train或val文件夹中的图片路径
        corrupted_images_count = 0  # 统计损坏图片数量

        for subtask_path in subtasks_paths:
            if os.path.isdir(subtask_path):  # 确保是文件夹
                subfolders = os.listdir(subtask_path)  # 获取子任务目录下的所有文件和文件夹
                subfolders_paths = [os.path.join(subtask_path, subfolder) for subfolder in subfolders]

                for subfolder_path in subfolders_paths:
                    if os.path.isdir(subfolder_path):  # 确保是文件夹
                        target_folder_path = os.path.join(subfolder_path, data_split,
                                                          flag)  # 使用 data_split 选择 train 或 val
                        print(f"Entering folder: {subfolder_path}")
                        if os.path.exists(target_folder_path):
                            images = os.listdir(target_folder_path)
                            images_paths = [os.path.join(target_folder_path, image) for image in images if
                                            image.lower().endswith(tuple(extensions))]

                            # 统计损坏图片数量并移除损坏图片
                            valid_images_paths = []
                            skip_verify = True  # Default to False if not set

                            for image_path in tqdm(images_paths, desc="Verifying images", leave=False):
                                try:
                                    if not skip_verify:  # Only verify images if skip_verify is False
                                        with Image.open(image_path) as img:
                                            img.verify()  # 验证图片是否损坏
                                    valid_images_paths.append(image_path)
                                except Exception:
                                    print(f"Corrupted image: {image_path}")
                                    corrupted_images_count += 1
                            all_images_paths.extend(valid_images_paths)

        print(f"Total number of valid images: {len(all_images_paths)}")
        print(f"Total number of corrupted images: {corrupted_images_count}")
        return all_images_paths

    return get_all_image_paths(root, flag, data_split,select_data_list)

def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths



class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        # root = args.data_path if is_train else args.eval_data_path
        root = args.data_path
        self.data_list = []
        # self.select_data_list = ['stable_diffusion_v_1_4','stable_diffusion_v_1_5']
        self.select_data_list = None
        # if'GenImage' in root and root.split('/')[-1] != 'train':
        #     file_path = root

        #     if '0_real' not in os.listdir(file_path):
        #         for folder_name in os.listdir(file_path):
                
        #             assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

        #             for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
        #                 self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
        #             for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
        #                 self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
        #     else:
        #         for image_path in os.listdir(os.path.join(file_path, '0_real')):
        #             self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
        #         for image_path in os.listdir(os.path.join(file_path, '1_fake')):
        #             self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        # else:

        #     for filename in os.listdir(root):

        #         file_path = os.path.join(root, filename)

        #         if '0_real' not in os.listdir(file_path):
        #             for folder_name in os.listdir(file_path):
                    
        #                 assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

        #                 for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
        #                     self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
        #                 for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
        #                     self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
        #         else:
        #             for image_path in os.listdir(os.path.join(file_path, '0_real')):
        #                 self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
        #             for image_path in os.listdir(os.path.join(file_path, '1_fake')):
        #                 self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        if 'GenImage' in root:
            # Use GenImage dataset
            self.root = root
            real_img_list = loadpathslist_genimage(self.root, 'nature',
                                                   data_split="train" if is_train else "val",select_data_list=self.select_data_list)
            real_label_list = [0 for _ in range(len(real_img_list))]
            for img_path, label in zip(real_img_list, real_label_list):
                self.data_list.append({"image_path": img_path, "label": label})
            fake_img_list = loadpathslist_genimage(self.root, 'ai',
                                                   data_split="train" if is_train else "val",select_data_list=self.select_data_list)
            fake_label_list = [1 for _ in range(len(fake_img_list))]
            for img_path, label in zip(fake_img_list, fake_label_list):
                self.data_list.append({"image_path": img_path, "label": label})
        self.patch_size = 256


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
            # 获取image 的size
            # width, height = image.size
            # if width != 512 and height !=512:
            #     print("The width and height is ", width, height)
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        
        image = transform_mil_train_plus(image)
        patch_size = self.patch_size

        try:
            # split image to patches
            C, H, W = image.shape
            patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            # (C, 2, 2, 256, 256)

            # Rearrange to (num_patches, C, patch_size, patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

    
        x = patches
        # 创建和 patch 数量相同的 label 列表或 Tensor
        target_repeat = torch.tensor([targets] * x.shape[0])
        return x,target_repeat
        

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        self.patch_size = 256
        root = args.data_path if is_train else args.eval_data_path
    
        self.data_list = []
        # self.select_data_list = ['stable_diffusion_v_1_4','stable_diffusion_v_1_5']
        self.select_data_list = None
        if 'GenImage' in root:
            # Use GenImage dataset
            self.root = root
            real_img_list = loadpathslist_genimage(self.root, 'nature',
                                                   data_split="train" if is_train else "val",select_data_list=self.select_data_list)
            real_label_list = [0 for _ in range(len(real_img_list))]
            for img_path, label in zip(real_img_list, real_label_list):
                self.data_list.append({"image_path": img_path, "label": label})
            fake_img_list = loadpathslist_genimage(self.root, 'ai',
                                                   data_split="train" if is_train else "val",select_data_list=self.select_data_list)
            fake_label_list = [1 for _ in range(len(fake_img_list))]
            for img_path, label in zip(fake_img_list, fake_label_list):
                self.data_list.append({"image_path": img_path, "label": label})
        
        if 'AIGCDetect_testset' in root:
                # 如果opt中的dataroot不为空，则使用该路径加载数据
                self.root = root
                real_img_list = loadpathslist(self.root,'0_real')    
                # 加载真实图像的路径列表
                real_label_list = [0 for _ in range(len(real_img_list))]
                # 为真实图像创建标签列表，标签为0
                fake_img_list = loadpathslist(self.root,'1_fake')
                fake_label_list = [1 for _ in range(len(fake_img_list))]
                for img_path, label in zip(real_img_list, real_label_list):
                    self.data_list.append({"image_path": img_path, "label": label})
                for img_path, label in zip(fake_img_list, fake_label_list):
                    self.data_list.append({"image_path": img_path, "label": label})
                

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        image = transform_mil_test(image)   
        patch_size = self.patch_size

        try:
            # split image to patches
            C, H, W = image.shape
            patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            # (C, 2, 2, 256, 256)

            # Rearrange to (num_patches, C, patch_size, patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        x = patches
        # 创建和 patch 数量相同的 label 列表或 Tensor
        target_repeat = torch.tensor([targets] * x.shape[0])
        return x, target_repeat

