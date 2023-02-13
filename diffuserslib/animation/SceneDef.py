

class DiffusionParams():
    def __init__(self, type="img2img", prompt:str="", negprompt:str="", strength:float=0.1, cfgscale:float=9, scheduler:str=None, model:str=None):
        self.type = type
        self.prompt= prompt
        self.negprompt = negprompt
        self.strength = strength
        self.cfgscale = cfgscale
        self.scheduler = scheduler
        self.model = model


class TransformParams():
    def __init__(self, type, timing="Linear", **kwargs):
        self.type = type
        self.timing = timing
        self.params = kwargs
        

class Scene():
    def __init__(self, name: str, initimage, length: int, diffusion: DiffusionParams, transform: TransformParams):
        self.name = name
        self.initimage = initimage
        self.keepinit = True           # specify whether to keep init image as first frame
        self.length = length
        self.diffusion = diffusion
        self.transform = transform

