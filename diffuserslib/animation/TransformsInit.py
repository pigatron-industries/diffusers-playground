from .Transforms import Transform
from .Interpolation import LinearInterpolation
from ..inference import DiffusersPipelines
from ..geometry import GraphicsContext
from typing import List
from PIL import Image


class InitTransform(Transform):
    def __init__(self, length, width, height, interpolation=LinearInterpolation(), transforms:List[Transform]=None, **kwargs):
        super().__init__(length, interpolation)
        self.transforms = transforms
        self.context = GraphicsContext((width, height))


class InitImageTransform(InitTransform):
    def __init__(self, length, inputdir:str, image:str, transforms:List[Transform]=None, **kwargs):
        self.image = Image.open(f"{inputdir}/{image}")
        super().__init__(length=length, width=self.image.width, height=self.image.height, transforms=transforms)
        self.filename = image

    def transform(self, image):
        currentframe = self.image
        if self.transforms is not None:
            for transform in self.transforms:
                    currentframe = transform(currentframe)
            self.image = currentframe
        return self.image


class InitTextToImageTransform(InitTransform):
    def __init__(self, pipelines:DiffusersPipelines, length, prompt="", negprompt="", cfgscale=9, steps=0.1, scheduler=None, seed=None, **kwargs):
        super().__init__(length=length)
        self.pipelines = pipelines
        self.prompt = prompt
        self.negprompt = negprompt
        self.cfgscale = cfgscale
        self.steps = steps
        self.scheduler = scheduler
        self.seed = seed

    def transform(self, image):
        image, seed = self.pipelines.textToImage(inimage=image, prompt=self.prompt, negprompt=self.negprompt, steps=self.steps, scale=self.cfgscale, seed=self.seed, scheduler=self.scheduler)
        return image