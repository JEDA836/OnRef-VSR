# -*- coding: utf-8 -*-
import os
import shutil

# 设置源文件夹路径
source_folder = '/data1/fdz/dataset/REDS/train/gt_cut'  # 替换为你的文件夹路径

# 遍历子文件夹
for i in range(270):
    subfolder = os.path.join(source_folder, f"{i:03d}")  # 子文件夹路径
    if os.path.exists(subfolder):
        # 目标子文件夹路径
        target_subfolder = os.path.join(source_folder, f"{i + 270:03d}")
        os.makedirs(target_subfolder, exist_ok=True)  # 创建目标子文件夹
        
        # 复制后15张图片并重命名
        for j in range(15, 30):
            src_image_path = os.path.join(subfolder, f"000000{j}.png")
            target_image_path = os.path.join(target_subfolder, f"{str(j - 15).zfill(8)}.png")
            shutil.copy(src_image_path, target_image_path)

print("over")
