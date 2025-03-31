import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K
# --------------------------
# 配置参数
# --------------------------
seed = 42
num_samples = 100
input_dir = '/raid0/chr/AIGCD/Datasets/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai'  # <<< 替换为你真实的图片文件夹路径
output_dir = '/raid5/chr/AIGCD/AIDE/data/verify_results'

# --------------------------
# 随机种子
# --------------------------
random.seed(seed)
torch.manual_seed(seed)

# --------------------------
# 检查路径
# --------------------------
os.makedirs(output_dir, exist_ok=True)
all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_images = random.sample(all_images, min(num_samples, len(all_images)))
print(selected_images)
# 模拟 Perturbations 和 RandomMask
class Perturbations:
    def __init__(self):
        pass
    def __call__(self, x):
        return (x,)  # mock: 不做扰动，直接返回原图 Tensor

class RandomMask:
    def __init__(self, ratio=(0.00, 0.75), patch_size=16, p=0.5):
        self.ratio = ratio
        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor
        # 简化版：随机遮挡一定数量的 patch
        c, h, w = tensor.shape
        num_patches = int((h // self.patch_size) * (w // self.patch_size))
        num_mask = int(torch.randint(int(self.ratio[0]*num_patches), int(self.ratio[1]*num_patches)+1, (1,)))
        for _ in range(num_mask):
            x = torch.randint(0, w - self.patch_size, (1,)).item()
            y = torch.randint(0, h - self.patch_size, (1,)).item()
            tensor[:, y:y+self.patch_size, x:x+self.patch_size] = 0.0
        return tensor

# 尺寸处理逻辑
def resize_or_crop(img):
    w, h = img.size
    if w > 512 and h > 512:
        return transforms.RandomCrop((512, 512))(img)
    else:
        return transforms.Resize((512, 512))(img)

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

# 定义完整 transform
transform_mil_train = transforms.Compose([
    transforms.Lambda(resize_or_crop),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(180),
    # transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: Perturbations()(x)[0]),
    RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 反标准化函数用于显示
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

# --------------------------
# 开始处理每张图片
# --------------------------
print(11111)
for idx, filename in enumerate(selected_images):
    try:
        print(idx)
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        transformed = transform_mil_train(img)
        print(transformed.shape)
        # 反标准化 & 可视化保存
        aug_img = denormalize(transformed)

        # plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img.resize((512, 512)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(aug_img.permute(1, 2, 0))
        axes[1].set_title('Augmented')
        axes[1].axis('off')

        save_path = os.path.join(output_dir, f"{idx+1:03d}.jpg")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Failed on {filename}: {e}")