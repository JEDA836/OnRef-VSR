# -- coding:utf-8 --
import os
import cv2
import numpy as np

# ????????????��?��??
input_folder = '/data1/fdz/dataset/REDS/train/gt'
output_folder = '/data1/fdz/dataset/REDS/train/cubic'

# ???????????��???????????
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ????????????��???????????��???????
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # ??????
            image_path = os.path.join(root, filename)
            img = cv2.imread(image_path)

            # mean = 0
            # std_dev = 1.6
            # gaussian_noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)
            # noisy_image = img.astype(np.float32) + gaussian_noise
            # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            # ˫�����²����ı�
            downsampled_image = cv2.resize(img, 
                                        (img.shape[1] // 4, img.shape[0] // 4), 
                                        interpolation=cv2.INTER_CUBIC)

            # ??????��???
            relative_path = os.path.relpath(image_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            # ???????????��???????????
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(output_path)
            # ?????????????��???
            cv2.imwrite(output_path, downsampled_image)

print("over")
