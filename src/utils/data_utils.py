import numpy as np
from datasets import Dataset, Image
import os
from transformers import pipeline
import tqdm
from PIL import ImageEnhance


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
    snow_modifiers = ["", " and cold", " and snowy", " and very snowy"]
    caption_prompt = "A satellite image of the earth. The weather is {cloud}{snow}."
    captions = []
    cloud_segments = []
    snow_segments = []
    for path, subdirs, files in os.walk(src_dir):
        for name in files:
            cloud_mod, snow_mod = 0,0
            if name.endswith(".npy"):
                bit_image = np.load(os.path.join(path, name))
                cloud_mod = get_cloud_segment(bit_image)
                snow_mod = get_snow_segment(bit_image)
                caption = caption_prompt.format(cloud=cloud_modifiers[cloud_mod], snow=snow_modifiers[snow_mod])
                captions.append(caption)
                cloud_segments.append(cloud_mod)
                snow_segments.append(snow_mod)
    return captions, cloud_segments, snow_segments

def get_snow_segment_binary(bit_image):
    snow_pixels = (np.right_shift(bit_image,5)&1).sum()
    snow_ratio = snow_pixels / bit_image.size
    if snow_ratio < 0.1:
        return 0
    else:
        return 1

def get_cloud_segment_binary(bit_image):
    cloud_pixels = (np.right_shift(bit_image,3)&1).sum()
    cloud_ratio = cloud_pixels / bit_image.size
    if cloud_ratio < 0.2:
        return 0
    else:
        return 1

def get_binary_captions(src_dir):
    cloud_modifiers = ["clear",  "cloudy"]
    snow_modifiers = ["sunny", "snowy"]
    caption_prompt = "A satellite image of the earth. The weather is {cloud} and {snow}."
    captions = []
    cloud_segments_binary = []
    snow_segments_binary = []

    for path, subdirs, files in os.walk(src_dir):
        for name in files:
            cloud_mod, snow_mod = 0,0
            if name.endswith(".npy"):
                bit_image = np.load(os.path.join(path, name))
                cloud_mod = get_cloud_segment_binary(bit_image)
                snow_mod = get_snow_segment_binary(bit_image)
                caption = caption_prompt.format(cloud=cloud_modifiers[cloud_mod], snow=snow_modifiers[snow_mod])
                captions.append(caption)
                cloud_segments_binary.append(cloud_mod)
                snow_segments_binary.append(snow_mod)


    return captions, cloud_segments_binary, snow_segments_binary

def brighten_targets(data):
    data['target'] = ImageEnhance.Brightness(data['target']).enhance(2.6)
    return data


def make_hf_dataset(input_folder, target_folder, masks_folder, save_to_disk=True, dset_location=None):
    split_captions, cloud_segments, snow_segments = get_captions(masks_folder)
    binary_captions,cloud_segments_binary, snow_segments_binary = get_binary_captions(masks_folder)
    input_paths = [os.path.join(input_folder, path) for path in os.listdir(input_folder)]
    target_paths = [os.path.join(target_folder, path) for path in os.listdir(target_folder)]
    dataset = Dataset.from_dict({"input" : input_paths, "target" : target_paths, 
                                "captions" : split_captions, "binary_captions" : binary_captions,
                                "cloud_segments" : cloud_segments, "snow_segments" : snow_segments,
                                "cloud_segments_binary" : cloud_segments_binary, 
                                "snow_segments_binary" : snow_segments_binary}).cast_column("input", Image()).cast_column("target", Image())
    dataset = dataset.map(brighten_targets)
    if save_to_disk:
        print("Saving dset to disk...")
        assert dset_location is not None, "Please provide a location to save the dataset to (dset_location)"
        dataset.save_to_disk(dset_location)
    return dataset
