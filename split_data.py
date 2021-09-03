import os
import random
from shutil import copyfile
from tqdm.auto import tqdm

size = 32

os.makedirs(f"images_data{size}/train", exist_ok = True)
os.makedirs(f"images_data{size}/val", exist_ok = True)
os.makedirs(f"images_data{size}/test", exist_ok = True)

random.seed(0)
for root, dirs, files in tqdm(os.walk(f"images{size}")):
    for image_file in files:
        _, label = os.path.split(root)
        pick = random.random()
        if pick < 0.7:
            folder = f"images_data{size}/train/{label}"
        elif pick < 0.9:
            folder = f"images_data{size}/val/{label}"
        else:
            folder = f"images_data{size}/test/{label}"

        os.makedirs(folder, exist_ok = True)
        copyfile(os.path.join(root, image_file), os.path.join(folder, image_file))


