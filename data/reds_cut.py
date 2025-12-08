import os
from PIL import Image

# 源文件夹和目标文件夹路径
src_folder = '/data1/fdz/dataset/REDS/train/gt'
dst_folder = '/data1/fdz/dataset/REDS/train/gt_cut'

# 确保目标文件夹存在
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 遍历源文件夹中的每个子文件夹
for subdir in os.listdir(src_folder):
    subdir_path = os.path.join(src_folder, subdir)
    if os.path.isdir(subdir_path):
        # 创建对应的目标子文件夹
        dst_subdir_path = os.path.join(dst_folder, subdir)
        if not os.path.exists(dst_subdir_path):
            os.makedirs(dst_subdir_path)
        
        # 获取当前子文件夹中的前30张图片
        images = sorted(os.listdir(subdir_path))[:30]
        
        for image_name in images:
            img_path = os.path.join(subdir_path, image_name)
            with Image.open(img_path) as img:
                # 截取前256x256行列
                cropped_img = img.crop((0, 0, 256, 256))
                
                # 保存截取后的图片到目标子文件夹
                cropped_img.save(os.path.join(dst_subdir_path, image_name))

print("完成所有图片处理。")
