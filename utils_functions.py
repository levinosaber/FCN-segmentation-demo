import json
import os 

def load_img_preprocessing_config(image_preprocessing_config_path: str = "./image_preprocessing_config.json", file_extention: str = "json"):
    '''
    load image preprocessing config from json file
    now only json is supported
    '''
    if not os.path.exists(image_preprocessing_config_path):
        raise FileNotFoundError(f"the image preprocessing config file: {image_preprocessing_config_path} does not exists")
    if file_extention == "json":
        with open(image_preprocessing_config_path, "r", encoding="utf-8") as f:
            img_preprocessing_config = json.load(f)
        f.close()
        default_config = {
            "base_size": 520,
            "crop_size": 480,
            "hflip_prob": 0.5,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        for key, default_value in default_config.items():
            if key not in img_preprocessing_config:
                print("Warning: the image preprocessing config file that provided is not complete, use default value instead")
                img_preprocessing_config[key] = default_value
        return img_preprocessing_config
    else:
        raise ValueError(f"the file extention: {file_extention} is not supported")
    