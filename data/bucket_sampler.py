"""
bucket_sampler.py
Resolution‑aware Bucket Sampler for AI‑generated‑image detection
Author: Haoran – 2025‑04‑19
----------------------------------------------------------------
• TEMPLATE_SIZES：离散模板，用于把任意分辨率映射到 16 个 bucket
• PATCH_RES      ：单个 patch 的边长（像素）
• batch_patches  ：一个 batch 里希望容纳的 patch 数量
----------------------------------------------------------------

"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm
import os
from glob import glob
# ----------------------------------------------------------------------
# 1. 全局常量
# ----------------------------------------------------------------------
TEMPLATE_SIZES: Sequence[Tuple[int, int]] = [
    (256, 256), (256, 512), (256, 768), (256, 1024),
    (512, 256), (512, 512), (512, 768), (512, 1024),
    (768, 256), (768, 512), (768, 768), (768, 1024),
    (1024, 256), (1024, 512), (1024, 768), (1024, 1024),
]

PATCH_RES = 256                 # 单个 patch 的边长
MAX_PATCHES_PER_IMG = 16        # 1024×1024 模板情况下最多 16 个 patch


# ----------------------------------------------------------------------
# 2. 工具函数
# ----------------------------------------------------------------------

def choose_best_template(
    h: int,
    w: int,
    templates: Sequence[Tuple[int, int]] = TEMPLATE_SIZES
) -> Tuple[int, int]:
    """
    1) 先按长宽比差选出最接近的 template 子集  
    2) 再在子集中按 (|th-h| + |tw-w|) 最小选出最终模板
    """
    target_ratio = w / h
    # print('The target_ratio is {}'.format(target_ratio))

    # 第一步：计算每个模板的比例差，并找出最小值
    ratio_dists = [abs((tw/th) - target_ratio) for th, tw in templates]
    min_ratio_dist = min(ratio_dists)
    # 筛出「比例差 == 最小」的模板列表
    candidates = [
        tpl for tpl, rd in zip(templates, ratio_dists)
        if rd == min_ratio_dist
    ]

    # 第二步：在 candidates 中，按 L1 尺寸差选最小
    def size_dist(tpl: Tuple[int, int]) -> int:
        th, tw = tpl
        return abs(th - h) + abs(tw - w)

    best_tpl = min(candidates, key=size_dist)
    return best_tpl
def patch_count_from_template(template: Tuple[int, int],
                              patch_res: int = PATCH_RES) -> int:
    th, tw = template
    return (th // patch_res) * (tw // patch_res)


def get_image_hw(path: str | Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.height, img.width


# ----------------------------------------------------------------------
# 3. 统计器（可选，用于调试采样分布）
# ----------------------------------------------------------------------
class BucketStats:
    def __init__(self):
        self.img_counter = Counter()        # bucket_id → #images
        self.sample_counter = Counter()     # bucket_id → #samples used in batches
        self.patch_counter = Counter()      # bucket_id → total patches used
        self.label_counter = defaultdict(Counter)  # bucket_id → label → count

    def add_image(self, bucket_id: int):
        self.img_counter[bucket_id] += 1

    def add_sample(self, bucket_id: int, patch_cnt: int, label: int = None):
        self.sample_counter[bucket_id] += 1
        self.patch_counter[bucket_id] += patch_cnt
        if label is not None:
            self.label_counter[bucket_id][label] += 1

    def summary(self) -> Dict[str, Dict[int, float]]:
        total_samples = sum(self.sample_counter.values()) or 1
        total_patches = sum(self.patch_counter.values()) or 1

        sample_probs = {k: v / total_samples for k, v in self.sample_counter.items()}
        patch_probs = {k: v / total_patches for k, v in self.patch_counter.items()}

        return {
            "images_per_bucket": dict(self.img_counter),
            "samples_used": dict(self.sample_counter),
            "patches_sampled": dict(self.patch_counter),
            "sample_prob": sample_probs,
            "patch_prob": patch_probs,
            "label_dist": {k: dict(v) for k, v in self.label_counter.items()}
        }

    def pretty_print(self) -> None:
        s = self.summary()
        header = f"{'Bucket':>6} | {'#Images':>8} | {'#Samples':>8} | {'#Patches':>9} | {'Sample%':>8} | {'Patch%':>8} | Label Dist"
        print(header)
        print("-" * len(header))
        for k in range(1, MAX_PATCHES_PER_IMG + 1):
            imgs = s["images_per_bucket"].get(k, 0)
            samples = s["samples_used"].get(k, 0)
            patches = s["patches_sampled"].get(k, 0)
            sample_prob = s["sample_prob"].get(k, 0) * 100
            patch_prob = s["patch_prob"].get(k, 0) * 100
            label_dist = s["label_dist"].get(k, {})
            label_str = ", ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(label_dist.items()))
            print(f"{k:>6} | {imgs:>8} | {samples:>8} | {patches:>9} | {sample_prob:>7.2f}% | {patch_prob:>7.2f}% | {label_str}")

# ----------------------------------------------------------------------
# 4. 核心 – BucketSampler
# ----------------------------------------------------------------------
class BucketSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset,
        batch_patches: int = 128,
        patch_res: int = PATCH_RES,
        template_sizes: Sequence[Tuple[int, int]] = TEMPLATE_SIZES,
        high_res_boost: float = 1.0,
        boost_slope: float = 0.2,
        max_boost: float = 3.0,
        drop_last: bool = False,
        seed: int | None = None,
        rank: int = None,
        num_replicas: int = None,
        shuffle: bool = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_patches = batch_patches
        self.patch_res = patch_res
        self.high_res_boost = high_res_boost
        self.boost_slope = boost_slope
        self.max_boost = max_boost
        self.drop_last = drop_last
        self.rng = random.Random(seed + rank if seed is not None else None)
        self.shuffle = shuffle
        self.rank = rank if rank is not None else 0
        self.world_size = num_replicas if num_replicas is not None else 1

        self.bucket2indices: Dict[int, List[int]] = defaultdict(list)
        self.idx2patches: List[int] = [0] * len(dataset)
        self.stats = BucketStats()

        print("[BucketSampler] Building buckets …")
        for idx in tqdm(range(len(dataset)), unit="img"):
            try:
                h, w = get_image_hw(dataset.image_paths[idx])
            except:
                print("load image error:", dataset.image_paths[idx])
                continue
            tpl = choose_best_template(h, w, template_sizes)
            p_cnt = patch_count_from_template(tpl, patch_res)
            p_cnt = max(1, min(MAX_PATCHES_PER_IMG, p_cnt))
            self.idx2patches[idx] = p_cnt
            self.bucket2indices[p_cnt].append(idx)
            self.stats.add_image(p_cnt)

            self.stats.add_sample(p_cnt,p_cnt,label= 0 if 'nature' in dataset.image_paths[idx] else 1 )

        def linear_boost(patch_count):
            if patch_count <= 8:
                return 1.0
            boost = 1.0 + self.boost_slope * (patch_count - 8)
            return min(boost, self.max_boost)

        self.bucket_weights = {
            k: len(self.bucket2indices[k]) * linear_boost(k)
            for k in self.bucket2indices
        }

    def __iter__(self):
        bucket_pools = {
            k: self.rng.sample(v, len(v)) if self.shuffle else list(v)
            for k, v in self.bucket2indices.items()
        }
        lowres_pool = bucket_pools.get(1, [])
        all_bucket_ids = list(bucket_pools.keys())
        all_batches = []

        while True:
            candidate_buckets = [b for b in all_bucket_ids if bucket_pools[b]]
            if not candidate_buckets:
                break

            batch, patch_sum = [], 0
            self.rng.shuffle(candidate_buckets)

            for b in candidate_buckets:
                pool = bucket_pools[b]
                while pool:
                    idx = pool.pop()
                    patch_cnt = self.idx2patches[idx]
                    batch.append((idx, patch_cnt))
                    patch_sum += patch_cnt
                    if patch_sum >= self.batch_patches:
                        break
                if patch_sum >= self.batch_patches:
                    break

            # 删除多余的 patch
            while patch_sum > self.batch_patches and batch:
                idx, cnt = batch.pop()
                patch_sum -= cnt

            # 补齐 patch=1 的图
            while patch_sum < self.batch_patches and lowres_pool:
                idx = lowres_pool.pop()
                batch.append((idx, 1))
                patch_sum += 1

            if patch_sum == self.batch_patches:
                all_batches.append([idx for idx, _ in batch])
            else:
                break

        for i, b in enumerate(all_batches):
            if i % self.world_size == self.rank:
                yield b

    def __len__(self) -> int:
        total_patch = sum(self.idx2patches)
        return (total_patch // self.batch_patches if self.drop_last
                else math.ceil(total_patch / self.batch_patches))


# ----------------------------------------------------------------------
# 5. ⛳ 示例用法
# ----------------------------------------------------------------------
def collect_image_paths(root):
    """
    只进入 class_x 是目录的子项（排除 zip/z01 等压缩文件），
    然后收集 train/val/nature/fake 下的所有 .jpg 图片路径
    """
    image_extensions = [
        '*.jpg',  '*.jpeg',  '*.png',  '*.bmp','*.JPEG','*.JPG','*.PNG','*.BMP']
    image_paths = []
    for datasets_name in os.listdir(root): # 不同的generator生成数据
        class_path = os.path.join(root, datasets_name)        
        for name in os.listdir(class_path):
            if not os.path.isdir(os.path.join(class_path,name)):  # ⚠️ 跳过不是目录的项
                continue
            _path = os.path.join(class_path,name)
            
            for split in ['train', 'val']:
                for label in ['nature', 'ai']:
                    for ext in image_extensions:
                        search_path = os.path.join(_path, split, label, ext)
                        image_paths.extend(glob(search_path))
    return image_paths
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    class DummyDataset:
        """只做示例：假设 image_paths 已经收集好"""
        def __init__(self, image_paths: List[str]):
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            # 实际任务里返回 (image, label) 等
            return self.image_paths[idx]

    # 假设手头已有 image list
    all_imgs = collect_image_paths(root='/raid0/chr/AIGCD/Datasets/GenImage')
    print('The totall number of images is: ',len(all_imgs))
    # print(all_imgs)
    ds = DummyDataset(all_imgs)

    sampler = BucketSampler(ds, batch_patches=128, high_res_boost=1.5, seed=42)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=4)

    print(f"Total batches per epoch ≈ {len(sampler)}")

    # 训练循环示例
    for epoch in range(1):
        for batch in tqdm(dl, desc=f"Epoch {epoch}"):
            # 在这里使用 batch (list of indices) 去加载图像、做 transform 等
            pass

    # 结束后打印统计
    print("\n📊  Bucket Sampling Summary")
    sampler.stats.pretty_print()