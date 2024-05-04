import os

from torch.utils.data import Dataset
from PIL import Image


def cat_list(images, fill_value=0):
    '''
    transform tuple images(a batch) to tensor
    '''
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class VOCSegmentation(Dataset):
    def __init__(self, voc_root, year = "2012", img_transforms = None, txt_name: str = "train.txt") -> None:
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be 2007 or 2012"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), f"the VOC dataset path: {root} is not exists"
        image_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "SegmentationClass")

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), f"the txt file: {txt_path} does not exists"

        # read txt file
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
        self.images_paths = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.mask_paths = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images_paths) == len(self.mask_paths)), "the number of images and masks must be equal"

        self.img_transforms = img_transforms

    def __getitem__(self, index):
        '''
        params:
            index (int): int index of the data
        return:
            tuple: (image, target), the target is the image segementation mask which is a image from VOCdevkit/VOC2012/SegmentationClass
        '''
        img = Image.open(self.images_paths[index]).convert("RGB")
        target = Image.open(self.mask_paths[index])

        if self.img_transforms is not None:
            img, target = self.img_transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.images_paths)
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)   # 将tuple形式的images转换为tensor.  transform tuple images to tensor
        batched_targets = cat_list(targets, fill_value=255)   # 将tuple形式的images转换为tensor
        return batched_imgs, batched_targets




# test this script

# voc_dataset = VOCSegmentation(voc_root = "./", year = "2012")
# d1 = voc_dataset[0]
# print(d1)