# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset
import traceback
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


# added by haoran
from tqdm import tqdm
from .patch_splitter import PatchifyImage
from .utils import loadpathslist,loadpathslist_genimage
from .transforms import transform_patch_based_test,transform_patch_based_train # CNNSpot的方法
from .transforms import transform_mil_train_fix, transform_mil_test_fix # MIL的方法 固定resize到512
from .transforms import transform_mil_train, transform_mil_test # MIL的方法
from .transforms import transform_mil_anyres_test # ONLY MIL EVAL的方法
from .transforms import transform_cnnspot_train,transform_cnnspot_test,transform_cnnspot_test_noresize # 100%复刻CNNSpot的方法
from collections import Counter
from .mil_datasets import TemplateMatcher
#################


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        root = args.data_path
        self.patch_res = 256
        self.data_mode = args.data_mode 
        self.bucket_sampler = args.bucket_sampler
        self.transforms = None
        self.mil_eval_mode = args.mil_eval_mode
        self.matcher = TemplateMatcher()
        
        if self.data_mode == 'cnnspot':
            # self.transforms = transform_patch_based_train
            self.transforms = transform_cnnspot_train
        elif self.data_mode == 'mil':
            if self.mil_eval_mode == 'only_eval':
                raise NotImplementedError
            elif self.bucket_sampler:
                self.transforms = transform_mil_train
            else:
                self.transforms = transform_mil_train_fix
        else:
            raise NotImplementedError
        
        self.data_list = []
        if 'GenImage' in root:
            # self.select_data_list = ['stable_diffusion_v_1_4'] # HACK
            self.select_data_list = args.select_data_list
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
        
        if 'CNNSpot_trainingdata' in root:
            # Use CNNSpot dataset
            self.root = root
            real_img_list = loadpathslist(self.root,'0_real')    
            # 加载真实图像的路径列表
            real_label_list = [0 for _ in range(len(real_img_list))]
            # 为真实图像创建标签列表，标签为0
            fake_img_list = loadpathslist(self.root,'1_fake')
            fake_label_list = [1 for _ in range(len(fake_img_list))]
            self.img = real_img_list + fake_img_list
            self.label = real_label_list + fake_label_list
            for img_path, label in zip(self.img, self.label):
                self.data_list.append({"image_path": img_path, "label": label})
            label_counter = Counter(self.label)
            num_real = label_counter.get(0, 0)  # 真实图像数量（label=0）
            num_fake = label_counter.get(1, 0)  # 伪造图像数量（label=1）

            print(f"真实样本数量（label=0）: {num_real}")
            print(f"伪造样本数量（label=1）: {num_fake}")
        self.image_paths = [item["image_path"] for item in self.data_list]  # 🔧 添加这一行
                    
    def __get_datalist__(self):
        return self.data_list
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
            if self.bucket_sampler:
                best_template = self.matcher.match_pil(image)
                image = self.matcher.resize_and_crop(image)
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        image = self.transforms(image) # (C, H, W)

        if self.data_mode == 'mil':
            patches = image.unfold(1, self.patch_res, self.patch_res).unfold(2, self.patch_res, self.patch_res)
            # print("patches shape:",patches.shape)
            # patches shape: (1, C, n_h, n_w, patch_res, patch_res)
            # 4) 重新排列成 (num_patches, C, patch_res, patch_res)
            n_h, n_w = patches.size(1), patches.size(2)
            patch_num = n_h*n_w
            patches = patches.permute(1, 2, 0, 3, 4).reshape(patch_num, 3, self.patch_res, self.patch_res)
            targets = torch.tensor([targets] * patch_num)
            if self.bucket_sampler: # 为了支持bucket sampling的
                return patches, targets, patch_num
            else:
                return patches, targets
            # return patches, targets, patch_num # 为了支持bucket sampling的
        else:
            pass
        return image, targets
        

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        root = args.data_path if is_train else args.eval_data_path
        self.data_mode = args.data_mode
        self.patch_res = 256
        self.transforms = None
        self.matcher = TemplateMatcher()
        self.bucket_sampler = args.bucket_sampler
        self.mil_eval_mode = args.mil_eval_mode
        if self.data_mode == 'cnnspot':
            # self.transforms = transform_patch_based_test
            self.transforms = transform_cnnspot_test if not args.no_crop else transform_cnnspot_test_noresize
        elif self.data_mode == 'mil':
            if self.mil_eval_mode == 'only_eval': # 只是在训练的时候使用MIL EVAL
                self.transforms = transform_mil_anyres_test
            elif self.bucket_sampler: # 使用bucket sampling
                self.transforms = transform_mil_test
            else:
                self.transforms = transform_mil_test_fix
        else:
            NotImplementedError
            
        self.select_data_list = args.select_data_list
        self.data_list = []
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
        if 'AIGCDetect_testset' in root or 'CNNSpot_trainingdata' in root or 'CNNSpot_val' in root:
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
            if self.bucket_sampler: # bucket sampling
                best_template = self.matcher.match_pil(image)
                image = self.matcher.resize_and_crop(image)
        except Exception as e:
            print(f"[Image Error] Path: {image_path}")
            traceback.print_exc()  # 打印完整错误堆栈
            print(f"Retrying with a random index...")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        image = self.transforms(image) # (C, H, W)

        if self.data_mode == 'mil':
            patches = image.unfold(1, self.patch_res, self.patch_res).unfold(2, self.patch_res, self.patch_res)
            # print("patches shape:",patches.shape)
            # patches shape: (1, C, n_h, n_w, patch_res, patch_res)
            # 4) 重新排列成 (num_patches, C, patch_res, patch_res)
            n_h, n_w = patches.size(1), patches.size(2)
            patch_num = n_h*n_w
            patches = patches.permute(1, 2, 0, 3, 4).reshape(patch_num, 3, self.patch_res, self.patch_res)
            targets = torch.tensor([targets] * patch_num)
            if self.bucket_sampler:
                return patches, targets, patch_num
            else:
                return patches, targets

        else:
            pass
        return image, targets
