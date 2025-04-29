import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from AIDE import HPF  # 确保这个是绝对导入 or 添加sys.path修复

def load_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: [1, 3, H, W]

def visualize_all_in_one(image_tensor, hpf_outputs, save_path="/raid5/chr/AIGCD/AIDE/chr_toolkits/visualization/hpf_visualization.png", vmax=None):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    outputs = hpf_outputs.squeeze(0).cpu().numpy()  # [30, H, W]

    total = 36  # 1原图 + 30个滤波图 + 5个空格
    cols = 6
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))

    # 展示原图
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # 展示滤波输出
    for idx in range(30):
        row = (idx + 1) // cols
        col = (idx + 1) % cols
        ax = axes[row, col]
        ax.imshow(outputs[idx], cmap='gray', vmax=vmax)
        ax.set_title(f"Filter {idx+1}")
        ax.axis('off')

    # 隐藏多余子图
    for idx in range(31, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved all-in-one visualization to: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "/raid5/chr/AIGCD/AIGCDetectBenchmark/AIGCDetect_testset/test/ADM/1_fake/625_adm_174.PNG"
    image = load_image(image_path).to(device)

    hpf_model = HPF().to(device)
    hpf_model.eval()

    with torch.no_grad():
        hpf_output = hpf_model(image)

    visualize_all_in_one(image, hpf_output)

if __name__ == "__main__":
    main()