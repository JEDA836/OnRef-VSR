import os
import shutil

# Path to the source folder
source_folder = '/data1/fdz/dataset/REDS/train/gt_cubic_cut'

# Path to the destination folder
destination_folder = '/data1/fdz/dataset/REDS/train/cubic_cut_15'

# Get a list of subfolders in the source folder
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

# Iterate over each subfolder
for subfolder in subfolders:


    # Copy the first 15 images to the corresponding subfolder in the destination folder
    for i in range(15):
        # source_file = subfolder[i]
        source_file = os.path.join(subfolder, f'{i:08d}.png')
        destination_subfolder = os.path.join(destination_folder, os.path.basename(subfolder))
        os.makedirs(destination_subfolder, exist_ok=True)
        destination_file = os.path.join(destination_subfolder, os.path.basename(source_file))
        shutil.copy2(source_file, destination_file)