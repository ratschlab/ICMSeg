import os
import random
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.augmentations import get_transform

NClASSES_dict = {"lumbarspine": 6}
anatomy_dict = {"VerSe": "lumbarspine",}
image_depth_verse = 120
IMAGE_DEPTH_dict = {
                    "VerSe": image_depth_verse,
                    "VerSe_contrast": image_depth_verse,
                    }

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def normalize(data, min=None, max=None):
    if min is None:
        min = np.min(data)
    if max is None:
        max = np.max(data)

    if max == min:
        data = np.zeros_like(data)
    else:
        data = (data - min) / (max - min)
    data = np.clip(data, 0, 1)
    data = (255 * data).astype(np.uint8)
    return data


class MIDataset(Dataset):
    def __init__(
        self,
        domain: str,
        path: str,
        split: str,
        filter: bool = False,
        dataset: str = "lumbarspine",
        transformname: str = None,
        rgb_dim: int = 1,
        dataset_style: str= None,
    ):  
        self.rgb_dim = rgb_dim

        self.dataset = domain
        
        self.dataset_style = dataset_style 
        
        anatomy = dataset

        self.depth = IMAGE_DEPTH_dict[domain]

        self.transform_name = transformname

        self.transforms, self.affine_transforms = get_transform(transformname)

        if self.dataset.endswith('_contrast'):
            self.folder_path = f"{path}/{anatomy}/{self.dataset.split('_')[0]}" 
        else:
            self.folder_path = f"{path}/{anatomy}/{self.dataset}"

        if filter and (split == "train"):
            filter_suffix = "-filtered"
        else:
            filter_suffix = ""

        imgs_folder = f"{self.folder_path}/images/{split}{filter_suffix}/"     
        self.images = [
            f"{imgs_folder}/{f}"
            for f in sorted(os.listdir(imgs_folder)) if ".png" in f
        ]

        if (self.dataset.endswith('_contrast')) and (self.dataset_style is not None):
            self.images_style = []  # Initialize the list of lists
            # Split the dataset_style string into a list of styles
            styles = self.dataset_style.split('+')
            for style in styles:  # Iterate over each style
                folder_path_style = f"{path}/{anatomy}/{style}"
                imgs_folder = f"{folder_path_style}/images/{split}{filter_suffix}/"
                print(imgs_folder)
                images_for_style = [
                    f"{imgs_folder}/{f}"
                    for f in sorted(os.listdir(imgs_folder)) if ".png" in f
                ]
                self.images_style.append(images_for_style)

        masks_folder = f"{self.folder_path}/labels/{split}{filter_suffix}/"
        self.masks = [
            f"{masks_folder}/{f}"
            for f in sorted(os.listdir(masks_folder)) if "_labelTrainIds.png" in f
        ]
        assert len(self.images) == len(self.masks), "The number of images and labels should be the same."
        
        for i, m in zip(self.images, self.masks):
            # Extract the base filename without extension.
            image_name = os.path.basename(i).split('.')[0]
            mask_name = os.path.basename(m).split('.')[0]
            
            # Remove the specific substring '_labelTrainIds' from mask_name.
            mask_name = mask_name.replace("_labelTrainIds", "")
            
            # Now compare the modified mask_name with the image_name.
            if image_name != mask_name:
                raise ValueError(f"File names do not match: {i} and {m}")

        self.num_classes = NClASSES_dict[anatomy]


    def prepare_img(self, index: int, images_list: List[str], masks_list: List[str]):
        img = Image.open(images_list[index])
        masks = Image.open(masks_list[index])
        basename = os.path.basename(images_list[index])

        if self.rgb_dim == 3:
            img = img.convert("RGB")
        else:
            img = img.convert('L')
        
        if self.transform_name is None:            
            img = np.array(img)
        elif 'bigaug' in self.transform_name:
            img = np.array(img).astype(np.float32) / 255.0

        return img, np.array(masks, dtype=np.int64), basename

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, masks, basename = self.prepare_img(index, self.images, self.masks)
        
        if self.dataset.endswith('_contrast') and (self.dataset_style is not None):
            # Now, use the chosen style list to prepare the image
            images_style = self.prepare_img(index, random.choice(self.images_style), self.masks)[0]
        else:
            images_style = None
            
        if self.transform_name == 'bigaug':
            img, masks, images_style = self.affine_transforms(img, masks, images_style)
            if not (images_style is None):
                transformed_imgs = (self.transforms(img), self.transforms(images_style))
            else:
                transformed_imgs = self.transforms(img)
        else:
            transformed_imgs = self.transforms(img)
            data = self.affine_transforms(image=np.array(img), mask=masks)
            img, masks = data["image"], data["mask"]

        return transformed_imgs, masks.astype(int), basename

    def __len__(self) -> int:
        return len(self.masks)