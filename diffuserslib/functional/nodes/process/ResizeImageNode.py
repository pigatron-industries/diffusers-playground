from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import random


class ResizeImageNode(FunctionalNode):
    def __init__(self, image:ImageFuncType, size=SizeFuncType, name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("size", size, Tuple[int, int])
        
        
    def process(self, image:Image.Image, size:Tuple[int, int]) -> Image.Image:
        return image.resize(size)
