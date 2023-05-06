import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CutPaste(object):
    def __init__(self, transform=True, _type="binary"):

        """
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification
        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        """
        self.type = _type
        if transform:
            self.transform = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            )
        else:
            self.transform = None

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.
        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range
        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(
            0, org_h - patch_h
        )
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch = transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(
            0, org_h - patch_h
        )
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)

        return aug_image

    def cutpaste_seg(
        self,
        image,
        area_ratio=(0.002, 0.01),
        # area_ratio=(0.02, 0.15),
        aspect_ratio=((0.3, 1), (1, 3.3)),
    ):
        """
        CutPaste augmentation
        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio
        :return: PIL image after CutPaste transformation
        """

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice(
            [random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])]
        )
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        cutpaste = self.crop_and_paste_patch(
            image, patch_w, patch_h, self.transform, rotation=False
        )
        return cutpaste

    def cutpaste_scar(self, image, width=[2, 16], length=[10, 25], rotation=(-45, 45)):
        """
        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation
        :return: PIL image after CutPaste-Scare transformation
        """
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = self.crop_and_paste_patch(
            image, patch_w, patch_h, self.transform, rotation=rotation
        )
        return cutpaste_scar

    def __call__(self, image):
        """
        :image: [PIL] - original image
        :return: if type == 'binary' returns original image and randomly chosen transformation, else it returns
                original image, an image after CutPaste transformation and an image after CutPaste-Scar transformation
        """
        if self.type == "segment":
            cutpaste_img = image.copy()
            # TODO: 回数を引数に出す
            for i in range(10):
                cutpaste_img = self.cutpaste_seg(cutpaste_img)
                cutpaste_img = self.cutpaste_scar(cutpaste_img)

            ano_img = np.array(image) - np.array(cutpaste_img)
            ano_img[ano_img != 0] = 255
            ano_img = Image.fromarray(ano_img)
            ano_img = ImageOps.grayscale(ano_img)
            return cutpaste_img, ano_img


class CutPasteDataset(Dataset):
    def __init__(
        self,
        dataset_path=None,
        image_size=(256, 256),
        mode="train",
    ):
        self.mode = mode
        self.dataset_path = dataset_path
        self.images, self.y, self.mask = self.load_dataset_folder()
        self.cutpaste_transform = CutPaste(_type="segment")

        if type(image_size) is not tuple:
            image_size = (image_size, image_size)

        self.crop_size = (224, 224)

        self.common_transform = transforms.Compose(
            [
                transforms.Resize(image_size, Image.ANTIALIAS),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
                transforms.CenterCrop(235),
                transforms.RandomCrop(self.crop_size),
            ]
        )

        self.transform_cutpaste_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_cutpaste_mask = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(image_size, Image.ANTIALIAS),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(image_size, Image.NEAREST),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.mode == "train":
            image_path = self.images[item]
            image = Image.open(image_path).convert("RGB")
            mask = torch.zeros([1, self.crop_size[0], self.crop_size[1]])
            outputs = self.cutpaste_transform(image)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            cutpaste_img = self.common_transform(outputs[0])
            cutpaste_img = self.transform_cutpaste_img(cutpaste_img)

            random.seed(seed)
            torch.manual_seed(seed)
            cutpaste_mask = self.common_transform(outputs[1])
            cutpaste_mask = self.transform_cutpaste_mask(cutpaste_mask)

            return (
                cutpaste_img,
                cutpaste_mask,
            )
        else:
            image_path = self.images[item]
            image = Image.open(image_path).convert("RGB")
            y = self.y[item]
            image = self.transform_test(image)

            if y == 0:
                mask = torch.zeros([1, self.crop_size[0], self.crop_size[1]])
            else:
                mask = self.mask[item]
                mask = Image.open(mask)
                mask = self.transform_mask(mask)

            return image, y, mask, image_path

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.mode)
        gt_dir = os.path.join(self.dataset_path, "ground_truth")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"

        return list(x), list(y), list(mask)
