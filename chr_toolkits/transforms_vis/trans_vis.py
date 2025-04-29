import os
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import kornia.augmentation as K

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
# 创建输出文件夹
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 反归一化
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    return tensor * std + mean

# 将 Tensor 保存为图像
def save_tensor_as_image(tensor, path):
    npimg = tensor.permute(1, 2, 0).numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imsave(path, npimg)



transform_patch_based_train = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 主逻辑
def main(image_dir, transform, save_dir, seed=42, sample_num=100):
    random.seed(seed)
    ensure_dir(save_dir)

    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    sampled_images = random.sample(all_images, sample_num)

    for idx, img_path in enumerate(sampled_images):
        img = Image.open(img_path).convert("RGB")
        x = transform(img)
        x = denormalize(x, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        save_path = os.path.join(save_dir, f"augmented_{idx}.png")
        save_tensor_as_image(x, save_path)
        print(f"✅ 保存：{save_path}")

if __name__ == "__main__":
    img_folder = "/raid0/chr/AIGCD/Datasets/CNNSpot_trainingdata/progan_train/airplane/1_fake"
    save_dir = "/raid5/chr/AIGCD/AIDE/chr_toolkits/transforms_vis/vis_results"
    main(img_folder, transform_patch_based_train, save_dir)