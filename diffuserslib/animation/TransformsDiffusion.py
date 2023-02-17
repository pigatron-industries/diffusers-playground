from .Transforms import Transform
from .Interpolation import InterpolationFunction, LinearInterpolation
from ..inference import DiffusersPipelines, compositedInpaint
from ..ImageUtils import alphaToMask


class ImageToImageTransform(Transform):
    def __init__(self, pipelines:DiffusersPipelines, length, prompt="", negprompt="", cfgscale=9, strength=0.1, scheduler=None, seed=None, **kwargs):
        super().__init__(length)
        self.pipelines = pipelines
        self.prompt = prompt
        self.negprompt = negprompt
        self.cfgscale = cfgscale
        self.strength = strength
        self.scheduler = scheduler
        self.seed = seed

    def transform(self, image):
        image, seed = self.pipelines.imageToImage(inimage=image, prompt=self.prompt, negprompt=self.negprompt, strength=self.strength, scale=self.cfgscale, seed=self.seed, scheduler=self.scheduler)
        return image


class OutpaintTransform(Transform):
    def __init__(self, pipelines:DiffusersPipelines, length, prompt="", negprompt="", cfgscale=9, steps=50, scheduler=None, seed=None, **kwargs):
        super().__init__(length)
        self.pipelines = pipelines
        self.prompt = prompt
        self.negprompt = negprompt
        self.cfgscale = cfgscale
        self.steps = steps
        self.scheduler = scheduler
        self.seed = seed

    def transform(self, image):
        maskimage = alphaToMask(image)
        image, seed = compositedInpaint(self.pipelines, initimage=image, maskimage=maskimage, prompt=self.prompt, negprompt=self.negprompt, steps=self.steps, scale=self.cfgscale, seed=self.seed, scheduler=self.scheduler)
        return image


class PromptInterpolationTransform(Transform):
    def __init__(self, pipelines:DiffusersPipelines, length, interpolation:InterpolationFunction=LinearInterpolation(), prompt="", negprompt="", cfgscale=9, steps=50, scheduler=None, seed=None, **kwargs):
        super().__init__(length, interpolation)
        self.pipelines = pipelines
        self.prompt = prompt
        self.negprompt = negprompt
        self.cfgscale = cfgscale
        self.steps = steps
        self.scheduler = scheduler
        self.seed = seed

    def transform(self, image):
        # TODO
        return image


class SeedInterpolationTransform(Transform):
    def __init__(self, pipelines:DiffusersPipelines, length, interpolation:InterpolationFunction=LinearInterpolation(), prompt="", negprompt="", cfgscale=9, steps=50, scheduler=None, seed=None, **kwargs):
        super().__init__(length, interpolation)
        self.pipelines = pipelines
        self.prompt = prompt
        self.negprompt = negprompt
        self.cfgscale = cfgscale
        self.steps = steps
        self.scheduler = scheduler
        self.seed = seed

    def transform(self, image):
        # TODO
        return image