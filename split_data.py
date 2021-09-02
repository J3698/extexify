import os
import random
from shutil import copyfile


os.makedirs("images_data/train", exist_ok = True)
os.makedirs("images_data/val", exist_ok = True)
os.makedirs("images_data/test", exist_ok = True)

random.seed(0)
for root, dirs, files in os.walk("images"):
    for image_file in files:
        _, label = os.path.split(root)
        pick = random.random()
        if pick < 0.7:
            folder = f"images_data/train/{label}"
        elif pick < 0.9:
            folder = f"images_data/val/{label}"
        else:
            folder = f"images_data/test/{label}"

        os.makedirs(folder, exist_ok = True)
        copyfile(os.path.join(root, image_file), os.path.join(folder, image_file))


