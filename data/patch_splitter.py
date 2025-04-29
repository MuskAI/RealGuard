import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
from torchvision.transforms.functional import to_tensor

class UnfoldPatchSplitter:
    def __init__(self, patch_size: int = 256):
        self.patch_size = patch_size

    def _auto_stride(self, length: int) -> Tuple[int, int]:
        P = self.patch_size
        min_stride = max(P // 2, 1)  # prevent overly small stride
        min_n = (length + P - 1) // P
        max_n = (length - P) // min_stride + 1

        for n in range(min_n, max_n + 1):
            if n <= 1:
                continue
            stride = (length - P) / (n - 1)
            if stride.is_integer() and stride >= min_stride:
                return int(stride), n

        # fallback if no suitable stride found
        stride = max((length - P) // (min_n - 1), 1) if min_n > 1 else P
        return stride, min_n

    def _pad_to_fit(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        B, C, H, W = images.shape
        stride_h, n_h = self._auto_stride(H)
        stride_w, n_w = self._auto_stride(W)

        total_h = (n_h - 1) * stride_h + self.patch_size
        total_w = (n_w - 1) * stride_w + self.patch_size

        pad_h = total_h - H
        pad_w = total_w - W

        images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return images_padded, (stride_h, stride_w), (pad_h, pad_w)

    def split(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
        B, C, H, W = images.shape
        images, (stride_h, stride_w), (pad_h, pad_w) = self._pad_to_fit(images)
        H_pad, W_pad = images.shape[-2:]

        patches = images.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        B_, nH, nW, C, P1, P2 = patches.shape
        patches = patches.view(-1, C, P1, P2)

        coords = []
        for i in range(nH):
            for j in range(nW):
                top = i * stride_h
                left = j * stride_w
                coords.append((top, left))

        coords = coords * B
        return patches, coords, (pad_h, pad_w), (stride_h, stride_w)

    def reconstruct_from_patches(self, patches: torch.Tensor, coords: List[Tuple[int, int]],
                                  image_shape: Tuple[int, int], batch_size: int) -> torch.Tensor:
        C = patches.shape[1]
        P = self.patch_size
        H, W = image_shape
        device = patches.device

        recon = torch.zeros((batch_size, C, H, W), device=device)
        count = torch.zeros((batch_size, 1, H, W), device=device)

        for idx, (top, left) in enumerate(coords):
            b = idx // (len(coords) // batch_size)
            recon[b, :, top:top+P, left:left+P] += patches[idx]
            count[b, :, top:top+P, left:left+P] += 1

        recon = recon / torch.clamp(count, min=1e-6)
        return recon

    def visualize(self, image: torch.Tensor, coords: List[Tuple[int, int]], pad: Tuple[int, int], stride: Tuple[int, int],index):
        import matplotlib.patches as patches
        import matplotlib.cm as cm

        image = image.cpu().permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        cmap = cm.get_cmap('hsv', len(coords))

        for i, (top, left) in enumerate(coords):
            color = cmap(i % len(coords))
            rect = patches.Rectangle((left, top), self.patch_size, self.patch_size,
                                     linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        pad_h, pad_w = pad
        stride_h, stride_w = stride
        plt.title(f"patch={self.patch_size}, stride=({stride_h}, {stride_w}), pad=({pad_h}, {pad_w})")
        plt.axis('off')
        # plt.show()
        plt.savefig(f"/raid5/chr/AIGCD/AIDE/data/tmp/{index}.png")


class PatchifyImage:
    def __init__(self, patch_size: int = 256):
        self.splitter = UnfoldPatchSplitter(patch_size=patch_size)

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.splitter.split(image)


# Example usage
if __name__ == "__main__":
    patchify = PatchifyImage(patch_size=256)
    splitter = patchify.splitter
    torch.manual_seed(42)

    for _ in range(20):
        h = random.randint(256, 256)
        w = random.randint(256, 1024)
        print(f"Testing image of size: {h}x{w}")
        image = torch.randn(3, h, w)
        patches, coords, pad, stride = patchify(image)
        print(f"{patches.shape=}, {len(coords)=}, pad={pad}, stride={stride}")
        splitter.visualize(image, coords[:len(coords)//1], pad, stride,_)
