from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List
import PIL
import os
import random
import numpy as np
import torch
import glob


PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


imagenet_object_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

subject_style_templates_small = [
    "a painting of a {subject} in the style of {style}",
    "a rendering of a {subject} in the style of {style}",
    "a cropped painting of a {subject} in the style of {style}",
    "the painting of a {subject} in the style of {style}",
    "a clean painting of a {subject} in the style of {style}",
    "a dirty painting of a {subject} in the style of {style}",
    "a dark painting of a {subject} in the style of {style}",
    "a picture of a {subject} in the style of {style}",
    "a cool painting of a {subject} in the style of {style}",
    "a close-up painting of a {subject} in the style of {style}",
    "a bright painting of a {subject} in the style of {style}",
    "a cropped painting of a {subject} in the style of {style}",
    "a good painting of a {subject} in the style of {style}",
    "a close-up painting of a {subject} in the style of {style}",
    "a rendition of a {subject} in the style of {style}",
    "a nice painting of a {subject} in the style of {style}",
    "a small painting of a {subject} in the style of {style}",
    "a weird painting of a {subject} in the style of {style}",
    "a large painting of a {subject} in the style of {style}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root:str,
        data_files:List[str],
        learnable_property="object",  # [object, style, subject_style]
        size=(512, 512),
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        subject = "girl"
    ):
        self.data_root = data_root
        self.data_files = data_files
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.subject = subject

        self.image_paths = []
        for filename in self.data_files:
            image_paths = glob.glob(f"{self.data_root}/{filename}")
            self.image_paths.extend(image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        if(learnable_property == "style"):
            self.templates = imagenet_style_templates_small
        elif(learnable_property == "object"):
            self.templates = imagenet_object_templates_small
        elif(learnable_property ==  "subject_style"):
            self.templates = subject_style_templates_small
            
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if(self.learnable_property == "subject_style"):
            text = random.choice(self.templates).format(subject = self.subject, style = self.placeholder_token)
        else:
            text = random.choice(self.templates).format(self.placeholder_token)
        example["caption"] = text

        if self.center_crop:
            # TODO
            pass

        if(self.size[1] is None):
            w, h = image.size
            image = image.resize((self.size[0], int(h * self.size[0] / w)), resample=self.interpolation)
        elif(self.size[0] is None):
            w, h = image.size
            image = image.resize((int(w * self.size[1] / h), self.size[1]), resample=self.interpolation)
        else:
            image = image.resize((self.size[0], self.size[1]), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example