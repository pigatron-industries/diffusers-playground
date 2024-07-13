from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from enum import Enum


class NoOpImageNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        
        
    def process(self, image:Image.Image|None) -> Image.Image|None:
        return image
