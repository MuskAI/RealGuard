import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

def load_image(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    return img

def compute_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 将低频移到中心
    magnitude = np.abs(fshift)
    log_spectrum = np.log1p(magnitude)  # 加1避免log(0)
    return log_spectrum

def visualize_spectrum(img, log_spectrum):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(log_spectrum, cmap='gray')
    plt.title("Log Amplitude Spectrum")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def crop_middle_frequency(log_spectrum, crop_size=64):
    h, w = log_spectrum.shape
    center_h, center_w = h // 2, w // 2
    cropped = log_spectrum[
        center_h - crop_size//2 : center_h + crop_size//2,
        center_w - crop_size//2 : center_w + crop_size//2
    ]
    plt.imshow(cropped, cmap='hot')
    plt.title("Middle Frequency Region")
    plt.axis('off')
    # plt.show()
    plt.savefig('/raid5/chr/AIGCD/AIDE/chr_toolkits/middle_frequency.png')

if __name__ == "__main__":
    image_path = '/raid5/chr/AIGCD/AIGCDetectionBenchMark/AIGCDetectionBenchMark/test/ADM/1_fake/625_adm_174.PNG'  # ← 替换为你的图像路径
    img = load_image(image_path)
    log_spectrum = compute_fft(img)
    visualize_spectrum(img, log_spectrum)
    crop_middle_frequency(log_spectrum)