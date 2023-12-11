import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
# Path to the folder containing your images
source_folder = "data/landsat_large"
input_folder = "inputs"
mask_folder = "masks"
target_folder = "outputs"
inputs = []
masks = []
targets = []
index = 0
for folder in os.listdir(source_folder):
    if not os.path.isdir(os.path.join(source_folder, folder)):
        continue
    for file in tqdm(os.listdir(os.path.join(source_folder, folder, input_folder))):
        if file.endswith(".jpg"):
            input_file = os.path.join(source_folder, folder, input_folder, file)
            mask_file = os.path.join(source_folder, folder, mask_folder, file.replace(".jpg", ".npy"))
            target_file = os.path.join(source_folder, folder, target_folder, file.replace(".jpg", ".jpg"))

            new_input = os.path.join(source_folder, folder, input_folder, str(index) + ".jpg")
            new_mask = os.path.join(source_folder, folder, mask_folder, str(index) + ".npy")
            new_target = os.path.join(source_folder, folder, target_folder, str(index) + ".jpg")
            os.rename(input_file, new_input)
            os.rename(mask_file, new_mask)
            os.rename(target_file, new_target)
            index += 1
            inputs.append(new_input)
            masks.append(new_mask)
            targets.append(new_target)

# If files already have unique names just do this part
# for folder in os.listdir(source_folder):
#     if not os.path.isdir(os.path.join(source_folder, folder)):
#         continue
#     for file in tqdm(os.listdir(os.path.join(source_folder, folder, input_folder))):
#         if file.endswith(".jpg"):
#             input_file = os.path.join(source_folder, folder, input_folder, file)
#             mask_file = os.path.join(source_folder, folder, mask_folder, file.replace(".jpg", ".npy"))
#             target_file = os.path.join(source_folder, folder, target_folder, file.replace(".jpg", ".jpg"))
#             inputs.append(input_file)
#             masks.append(mask_file)
#             targets.append(target_file)


# Shuffle everything together
data = list(zip(inputs, masks, targets))
random.shuffle(data)

# Create Train, Test, and Val folders if they don't exist
train_folder = os.path.join(source_folder, "train")
test_folder = os.path.join(source_folder, "test")
val_folder = os.path.join(source_folder, "val")

for folder in [train_folder, test_folder, val_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(os.path.join(folder, input_folder))
        os.makedirs(os.path.join(folder, mask_folder))
        os.makedirs(os.path.join(folder, target_folder))

# Calculate the number of images for each set (train, test, val)
total_images = len(data)
train_split = int(0.9 * total_images)
test_split = int(0.08 * total_images)
val_split = total_images - train_split - test_split


# Move images to their respective folders
try:
    for i, (input, mask, target) in tqdm(enumerate(data)):
        if i < train_split:
            # rename to index
            shutil.move(input, os.path.join(train_folder, input_folder, os.path.basename(input)))
            shutil.move(mask, os.path.join(train_folder, mask_folder, os.path.basename(mask)))
            shutil.move(target, os.path.join(train_folder, target_folder, os.path.basename(target)))


        elif i < train_split + test_split:
            shutil.move(input, os.path.join(test_folder, input_folder, os.path.basename(input)))
            shutil.move(mask, os.path.join(test_folder, mask_folder, os.path.basename(mask)))
            shutil.move(target, os.path.join(test_folder, target_folder, os.path.basename(target)))
        else:
            shutil.move(input, os.path.join(val_folder, input_folder, os.path.basename(input)))
            shutil.move(mask, os.path.join(val_folder, mask_folder, os.path.basename(mask)))
            shutil.move(target, os.path.join(val_folder, target_folder, os.path.basename(target)))
except:
    breakpoint()

print("Images have been divided into Train, Test, and Val folders.")