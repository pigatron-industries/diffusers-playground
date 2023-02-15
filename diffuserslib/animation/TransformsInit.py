from .Transforms import Transform
from ..geometry import GraphicsContext
from typing import List
from PIL import Image


class InitTransform(Transform):
    def __init__(self, length, interpolation, width, height, transforms:List[Transform], **kwargs):
        super().__init__(length, interpolation)
        self.transforms = transforms
        self.context = GraphicsContext((width, height))


class InitImageTransform(InitTransform):
    def __init__(self, length, inputdir:str, image:str, transforms:List[Transform], **kwargs):
        print(transforms)
        self.image = Image.open(f"{inputdir}/{image}")
        super().__init__(length=length, interpolation=None, width=self.image.width, height=self.image.height, transforms=transforms)
        self.filename = image

    def transform(self, image):
        currentframe = self.image
        if self.transforms is not None:
            for transform in self.transforms:
                    currentframe = transform(currentframe)
            self.image = currentframe
        return self.image