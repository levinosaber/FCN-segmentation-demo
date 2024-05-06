import os
import time
import datetime
import torch 

from torch.utils.data import DataLoader

from model.fcn import fcn_resnet50, fcn_resnet101
from train_utils import evaluate, train_one_epoch, create_lr_scheduler
from image_preprocessing.preprocessing_classes import get_img_transform
from Datasets_classes import VOCSegmentation
import image_preprocessing.img_transforms as T

def create_model(aux, num_classes, pretrain=True):
    model = fcn_resnet50(aux, num_classes)

    if pretrain:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main():
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 开发阶段用 参数输入
    config  = {
        "batch_size": 8,
        "num_classes": 21,

        "data_path": ".\\",
    }

    result_file_path = f"results{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    train_dataset = VOCSegmentation(config["data_path"], year="2012", img_transforms=get_img_transform(train_mode=True), txt_name="train.txt")
    val_dataset = VOCSegmentation(config["data_path"], year="2012", img_transforms=get_img_transform(train_mode=False), txt_name="val.txt")

    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, pin_memory= True, collate_fn=VOCSegmentation.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=VOCSegmentation.collate_fn)

    # # test train_loader
    # for data in train_loader:
    #     print(data)
    #     break

    # model = create_model(aux=args.aux, num_classes=num_classes)
    model = create_model(aux=True, num_classes=config["num_classes"], pretrain=True)
    model.to(device)


if __name__ == "__main__":
    main()