import os
import random
from shutil import copyfile
from tqdm.auto import tqdm

size = 32

os.makedirs(f"images_data{size}/train", exist_ok = True)
os.makedirs(f"images_data{size}/val", exist_ok = True)
os.makedirs(f"images_data{size}/test", exist_ok = True)

random.seed(0)
files = []
for root, dirs, files in tqdm(os.walk(f"images{size}")):
    random.shuffle(files)
    for image_file in files:
        _, label = os.path.split(root)
        pick = random.random()

        os.makedirs(f"images_data{size}/train/{label}", exist_ok = True)
        os.makedirs(f"images_data{size}/val/{label}", exist_ok = True)
        os.makedirs(f"images_data{size}/test/{label}", exist_ok = True)
        image_path = os.path.join(root, image_file)

        if len(os.listdir(f"images_data{size}/train/{label}")) == 0:
            copyfile(image_path, os.path.join(f"images_data{size}/train/{label}", image_file))
            continue
        elif len(os.listdir(f"images_data{size}/val/{label}")) == 0:
            copyfile(image_path, os.path.join(f"images_data{size}/val/{label}", image_file))
            continue
        elif len(os.listdir(f"images_data{size}/test/{label}")) == 0:
            copyfile(image_path, os.path.join(f"images_data{size}/test/{label}", image_file))
            continue

        if pick < 0.7:
            copyfile(image_path, os.path.join(f"images_data{size}/train/{label}", image_file))
        elif pick < 0.9:
            copyfile(image_path, os.path.join(f"images_data{size}/val/{label}", image_file))
        else:
            copyfile(image_path, os.path.join(f"images_data{size}/test/{label}", image_file))



