import os
from tqdm import tqdm

from glob import glob

def loadpathslist_genimage(root, flag, data_split="train",select_data_list=None):
    """
    @author: haoran
    @time: 2024/3/13 20:36
    @description: 从GenImage读取图片生成list
    """

    def get_all_image_paths(path, flag, data_split="train",select_data_list=None):
        """递归获取指定路径下的所有图片文件路径"""
        extensions = {".jpg", ".png", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"}

        subtasks = os.listdir(path)  # 获取path目录下的所有文件和文件夹
        # 如果subtasks中的内容在select_data_list里则选取，否则删除
        if select_data_list is not None:    
            subtasks = [subtask for subtask in subtasks if subtask in select_data_list]
        else:
            pass
        subtasks_paths = [os.path.join(path, task) for task in subtasks]  # 生成完整路径
        all_images_paths = []  # 存储所有train或val文件夹中的图片路径
        corrupted_images_count = 0  # 统计损坏图片数量

        for subtask_path in subtasks_paths:
            if os.path.isdir(subtask_path):  # 确保是文件夹
                subfolders = os.listdir(subtask_path)  # 获取子任务目录下的所有文件和文件夹
                subfolders_paths = [os.path.join(subtask_path, subfolder) for subfolder in subfolders]

                for subfolder_path in subfolders_paths:
                    if os.path.isdir(subfolder_path):  # 确保是文件夹
                        target_folder_path = os.path.join(subfolder_path, data_split,
                                                          flag)  # 使用 data_split 选择 train 或 val
                        print(f"Entering folder: {target_folder_path}")
                        if os.path.exists(target_folder_path):
                            images = os.listdir(target_folder_path)
                            images_paths = [os.path.join(target_folder_path, image) for image in images if
                                            image.lower().endswith(tuple(extensions))]

                            # 统计损坏图片数量并移除损坏图片
                            valid_images_paths = []
                            skip_verify = True  # Default to False if not set

                            for image_path in tqdm(images_paths, desc="Verifying images", leave=False):
                                try:
                                    if not skip_verify:  # Only verify images if skip_verify is False
                                        with Image.open(image_path) as img:
                                            img.verify()  # 验证图片是否损坏
                                    valid_images_paths.append(image_path)
                                except Exception:
                                    print(f"Corrupted image: {image_path}")
                                    corrupted_images_count += 1
                            all_images_paths.extend(valid_images_paths)

        print(f"Total number of valid images: {len(all_images_paths)}")
        print(f"Total number of corrupted images: {corrupted_images_count}")
        return all_images_paths

    return get_all_image_paths(root, flag, data_split,select_data_list)

def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths



