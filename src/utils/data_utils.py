import numpy as np
from datasets import Dataset, Image
import os
from transformers import pipeline
import tqdm


def get_cloud_segment(bit_image):
    cloud_pixels = (np.right_shift(bit_image,3)&1).sum()
    cloud_ratio = cloud_pixels / bit_image.size
    if cloud_ratio < 0.1:
        return 0
    elif cloud_ratio < 0.3:
        return 1
    elif cloud_ratio < 0.5:
        return 2
    else:
        return 3

def get_snow_segment(bit_image):
    snow_pixels = (np.right_shift(bit_image,5)&1).sum()
    snow_ratio = snow_pixels / bit_image.size
    if snow_ratio < 0.05:
        return 0
    elif snow_ratio < 0.1:
        return 1
    elif snow_ratio < 0.3:
        return 2
    else:
        return 3
    
def get_captions(src_dir):
    cloud_modifiers = ["clear", "slightly cloudy", "cloudy", "very cloudy"]
    snow_modifiers = ["", " and there is some snow", " and it is snowy", " and it is very snowy"]
    caption_prompt = "A satellite image of the earth. The weather is {cloud}{snow}."
    captions = []
    for path, subdirs, files in os.walk(src_dir):
        for name in files:
            cloud_mod, snow_mod = 0,0
            if name.endswith(".npy"):
                bit_image = np.load(os.path.join(path, name))
                cloud_mod = get_cloud_segment(bit_image)
                snow_mod = get_snow_segment(bit_image)
                caption = caption_prompt.format(cloud=cloud_modifiers[cloud_mod], snow=snow_modifiers[snow_mod])
                captions.append(caption)
    return captions


def make_hf_dataset(input_folder, target_folder, masks_folder, save_to_disk=True, dset_location=None):
    captions = get_captions(masks_folder)
    input_paths = [os.path.join(input_folder, path) for path in os.listdir(input_folder)]
    target_paths = [os.path.join(target_folder, path) for path in os.listdir(target_folder)]
    dataset = Dataset.from_dict({"input" : input_paths, "target" : target_paths, "caption" : captions}).cast_column("input", Image()).cast_column("target", Image())
    if save_to_disk:
        print("Saving dset to disk...")
        assert dset_location is not None, "Please provide a location to save the dataset to (dset_location)"
        dataset.save_to_disk(dset_location)
    return dataset
