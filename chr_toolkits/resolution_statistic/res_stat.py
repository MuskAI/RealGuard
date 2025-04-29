import os
from collections import Counter
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def collect_image_paths(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except:
        return None

def get_bucket(w, h):
    if w < 256 or h < 256:
        return "<256"
    elif w ==256 and h ==256:
        return "256"
    elif w < 512 or h < 512:
        return "257–512"
    elif w==512 and h==512:
        return "512"
    elif w < 1024 or h < 1024:
        return "513–1024"
    elif w==1024 and h==1024:
        return "1024"
    else:
        return ">1024"

def main(root_dir):
    image_paths = collect_image_paths(root_dir)
    bucket_counter = Counter()

    print(f"🔍 正在统计 {len(image_paths)} 张图片的尺寸分布...\n")
    for path in tqdm(image_paths, desc="处理进度"):
        res = get_image_resolution(path)
        if res:
            bucket = get_bucket(*res)
            bucket_counter[bucket] += 1

    total = sum(bucket_counter.values())

    print("\n📊 分辨率区间统计结果：\n")
    for bucket in ["<256","256", "257–512","512", "513–1024","1024", ">1024"]:
        count = bucket_counter.get(bucket, 0)
        percent = count / total * 100 if total > 0 else 0
        print(f"{bucket:10}: {count:4d} 张 ({percent:.2f}%)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python count_image_buckets.py <目录路径>")
    else:
        main(sys.argv[1])