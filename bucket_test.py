import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data.datasets_all import TrainDataset  # 替换为你dataset文件名
from data.bucket_sampler import BucketSampler  # 替换为你sampler文件名

# ====================
# 模拟 args
# ====================
class Args:
    def __init__(self):
        self.data_path = "/raid0/chr/AIGCD/Datasets/GenImage"  # 修改为你的路径
        self.data_mode = "mil"  # 'mil' 或 'cnnspot'
        self.loss_mode = "mil"
        self.select_data_list = None
        self.mil_eval_mode = False
        self.bucket_sampler = True
        self.seed = 42

args = Args()

# ====================
# 初始化 Dataset
# ====================
dataset = TrainDataset(is_train=True, args=args)
print("📂 Dataset length:", len(dataset))
# ====================
# 初始化 BucketSampler
# ====================
sampler = BucketSampler(
    dataset=dataset,
    batch_patches=64,     # 每个 batch 中 patch 数量
    rank=0,
    num_replicas=1,
    shuffle=True,
    seed=args.seed,
)


# ====================
# 自定义collate_fn
# ====================
def custom_collate_fn(batch):
    """
    每个样本是 (patches, labels, patch_num)
    patches: Tensor[num_patches, C, H, W]
    labels : Tensor[num_patches]
    """
    # 拆分批次中的每个字段
    pass
    images_list, labels_list, patch_nums = zip(*batch)

    # 拼接 patch：沿第 0 维 concat
    batch_images = torch.cat(images_list, dim=0)  # shape: (sum_patches, C, H, W)
    batch_labels = torch.cat(labels_list, dim=0)  # shape: (sum_patches,)

    return batch_images, batch_labels, torch.tensor(patch_nums)

# ====================
# 初始化 DataLoader
# ====================
loader = DataLoader(
    dataset,
    collate_fn=custom_collate_fn,
    batch_sampler=sampler,
    num_workers=4,
    pin_memory=True,
)
print(f"[Dataset] 样本数量: {len(dataset)}")
print(f"[Sampler] Patch 总数: {sum(sampler.idx2patches)}")
print(f"[Sampler] Batch 数量: {len(sampler)}")
# ====================
# 遍历一轮并输出信息
# ====================
print("🚀 开始遍历 BucketSampler:")
for i, batch in enumerate(loader):
    if args.data_mode == 'mil':
        images, targets, patch_num = batch
        print(f"所有标签: {targets.tolist()}")
    else:
        images, targets = batch
        print(f"Batch {i:03d} | Tensor Shape: {images.shape} | 标签: {targets}")

    if i >= 10:
        break  # 只展示前几个 batch

# ====================
# 输出统计分布
# ====================
print("\n📊 BucketSampler 统计分布:")
sampler.stats.pretty_print()