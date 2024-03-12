import numpy as np
import cv2
# A ref: https://albumentations.ai/docs/
import albumentations as A
from albumentations import (
    Compose,
    RandomResizedCrop,
    HorizontalFlip,
    RandomBrightnessContrast,
    Rotate,
)
from albumentations.pytorch.transforms import ToTensorV2


def data_augmenter(data_format, aug, crop=1, image_size=224):
    if data_format == "img":
        if aug == "default":
            augmenter = Compose([
                RandomResizedCrop(image_size, image_size, scale=(crop, 1.)),
                Rotate(border_mode=cv2.BORDER_CONSTANT),
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                ToTensorV2(),
            ])
        elif aug == "sham":
            augmenter = Compose([
                RandomResizedCrop(image_size, image_size, scale=(1., 1.)),
                ToTensorV2(),
            ])
    else:
        augmenter = None
    return augmenter