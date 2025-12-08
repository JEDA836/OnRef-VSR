import os
import shutil

# 定义文件夹路径
source_dir_A = '/data1/fdz/dataset/REDS/train/train_sharp_gauss_bilinear'  # 请替换为实际路径
source_dir_C = '/data1/fdz/dataset/REDS/val/val_sharp_gauss_bilinear'  # 请替换为实际路径
destination_dir_B = '/data1/fdz/dataset/REDS/train/gt_gauss_bilinear'  # 请替换为实际路径

# 获取A文件夹下所有文件夹，并按名称排序
folders_A = [f for f in os.listdir(source_dir_A) if os.path.isdir(os.path.join(source_dir_A, f))]
folders_A.sort()

# 从030开始递增命名
start_index = 30

# 创建目标文件夹B
if not os.path.exists(destination_dir_B):
    os.makedirs(destination_dir_B)

# 复制A文件夹中的文件夹并重命名
for folder in folders_A:
    new_folder_name = f"{start_index:03d}"
    shutil.copytree(os.path.join(source_dir_A, folder), os.path.join(destination_dir_B, new_folder_name))
    start_index += 1

# 获取C文件夹下所有文件夹
folders_C = [f for f in os.listdir(source_dir_C) if os.path.isdir(os.path.join(source_dir_C, f))]

# 复制C文件夹中的文件夹到目标文件夹B
for folder in folders_C:
    shutil.copytree(os.path.join(source_dir_C, folder), os.path.join(destination_dir_B, folder))

print("文件夹复制和重命名完成。")
