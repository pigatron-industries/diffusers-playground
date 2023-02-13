from typing import List


class TransformParams():
    def __init__(self, type, timing="Linear", **kwargs):
        self.type = type
        self.timing = timing
        self.params = kwargs
        

class Scene():
    def __init__(self, name: str, initimage, length: int, transforms:List[TransformParams]):
        self.name = name
        self.initimage = initimage
        self.keepinit = True           # specify whether to keep init image as first frame
        self.length = length
        self.transforms = transforms

