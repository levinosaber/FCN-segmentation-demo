import os
import time
import torch 
import json

import image_preprocessing.img_transforms as T
from utils_functions import load_img_preprocessing_config

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob = 0.5, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) -> None:
        min_size = int(base_size * 0.5)
        max_size = int(base_size * 2.0)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))  # do random horizontal flip
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean = mean, std = std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)
    

class SegmentationPresetVal:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
        self.transforms = T.Compose([
            T.RandomResize(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, img, target):
        return self.transforms(img, target)


def get_img_transform(train_mode: bool, image_preprocessing_config_path: str = "./image_preprocessing_config.json"):  
    ''' 
    get image transforms for training or validation, by default, the image_preprocessing_config_path is "./image_preprocessing_config.json"
    '''
    img_pre_config = load_img_preprocessing_config(image_preprocessing_config_path)  # second parameter not used
    base_size = img_pre_config["base_size"]
    crop_size = img_pre_config["crop_size"]

    return SegmentationPresetTrain(base_size, crop_size) if train_mode else SegmentationPresetVal(base_size)
        