
# bucket_sampler.py
# Resolution‑aware Bucket Sampler for AI‑generated‑image detection
# Author: Haoran – 2025‑04‑19
#
# 增加 TemplateMatcher 类，用于独立匹配输入图像到最合适的分辨率模板

from __future__ import annotations
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import os
from glob import glob
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm
import torchvision.transforms as T
# ----------------------------------------------------------------------
# 1. 全局常量
# ----------------------------------------------------------------------
TEMPLATE_SIZES: Sequence[Tuple[int, int]] = [
    (256, 256), (256, 512), (256, 768), (256, 1024),
    (512, 256), (512, 512), (512, 768), (512, 1024),
    (768, 256), (768, 512), (768, 768), (768, 1024),
    (1024, 256), (1024, 512), (1024, 768), (1024, 1024),
]
PATCH_RES = 256                 # 单个 patch 的边长（像素）
MAX_PATCHES_PER_IMG = 16        # 最多 16 个 patch (1024×1024 / 256²)

# ----------------------------------------------------------------------
# 2. 模板匹配类
# ----------------------------------------------------------------------
class TemplateMatcher:
    """
    独立的模板匹配器：
    - match_image(path) -> 返回 {height, width, template, patch_count}
    - match(h, w) -> 最优模板尺寸 (th, tw)
    - patch_count(template) -> 根据 patch_res 计算可拆分 patch 数量
    """
    def __init__(
        self,
        templates: Sequence[Tuple[int, int]] = TEMPLATE_SIZES,
        patch_res: int = PATCH_RES
    ):
        self.templates = templates
        self.patch_res = patch_res

    def get_image_hw(self, path: Union[str, Path]) -> Tuple[int, int]:
        """打开图像文件，返回 (height, width)"""
        with Image.open(path) as img:
            return img.height, img.width

    def match(self, h: int, w: int) -> Tuple[int, int]:
        """
        先按宽高比差筛选，再按尺寸 L1 距离找最优模板
        """
        target_ratio = w / h
        # 计算所有模板的比例差
        ratio_dists = [abs((tw/th) - target_ratio) for th, tw in self.templates]
        min_ratio = min(ratio_dists)
        # 保留比例差最小的候选模板
        candidates = [tpl for tpl, rd in zip(self.templates, ratio_dists) if rd == min_ratio]
        # 在候选中按 L1 尺寸差选最小
        def size_dist(tpl: Tuple[int, int]) -> int:
            th, tw = tpl
            return abs(th - h) + abs(tw - w)
        return min(candidates, key=size_dist)

    def patch_count(self, template: Tuple[int, int]) -> int:
        """根据模板尺寸计算能拆分出的 patch 数量"""
        th, tw = template
        return (th // self.patch_res) * (tw // self.patch_res)

    def match_image(self, path: Union[str, Path]) -> Dict[str, Union[int, Tuple[int, int]]]:
        """
        输入图像文件路径，输出匹配结果：
          height, width, template, patch_count
        """
        h, w = self.get_image_hw(path)
        tpl = self.match(h, w)
        p_cnt = max(1, min(MAX_PATCHES_PER_IMG, self.patch_count(tpl)))
        return {"height": h, "width": w, "template": tpl, "patch_count": p_cnt}

    def match_pil(self, img: Image.Image) -> Tuple[int, int]:
        """
        输入 PIL.Image 对象，返回最优模板尺寸 (th, tw)
        """
        # 获取 PIL.Image 的尺寸
        w, h = img.size  # img.size -> (width, height)
        # 调用已有匹配逻辑
        return self.match(h, w)
    
    def resize_and_crop(self, img: Image.Image) -> Image.Image:
        """
        将 PIL.Image 输入：
        1) 根据最佳模板尺寸 resize，保持原始长宽比例
        2) 使用 torchvision.transforms.CenterCrop 完成中心裁剪
        """
        tw, th = self.match_pil(img)
        src_w, src_h = img.size
        src_ratio = src_w / src_h
        tgt_ratio = tw / th
        # 按比例缩放，使短边至少满足目标尺寸
        if src_ratio > tgt_ratio:
            # 原图更宽：高度对齐
            new_h = th
            new_w = int(src_w * th / src_h)
        else:
            # 原图更高或相等：宽度对齐
            new_w = tw
            new_h = int(src_h * tw / src_w)
        # 先缩放
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        # 使用 torchvision 中心裁剪到目标尺寸 (th, tw)
        center_crop = T.CenterCrop((th, tw))
        return center_crop(img_resized)
    
    def split_patches(self, img: Image.Image) -> torch.Tensor:
        """
        将 PIL.Image 输入，自动 resize+crop，再使用 unfold 高效拆分成非重叠 patch
        返回 shape (num_patches, C, patch_res, patch_res)
        """
        # 1) Resize and crop to target template
        img_crop = self.resize_and_crop(img)
        # 2) 转为 Tensor, 添加 batch 维
        tensor = T.ToTensor()(img_crop).unsqueeze(0)  # (1, C, H, W)
        # 3) 使用 unfold 拆分
        patches = tensor.unfold(2, self.patch_res, self.patch_res).unfold(3, self.patch_res, self.patch_res)
        # patches shape: (1, C, n_h, n_w, patch_res, patch_res)
        # 4) 重新排列成 (num_patches, C, patch_res, patch_res)
        n_h, n_w = patches.size(2), patches.size(3)
        patches = patches.permute(2, 3, 1, 4, 5)
        return patches.reshape(-1, patches.size(2), self.patch_res, self.patch_res)
