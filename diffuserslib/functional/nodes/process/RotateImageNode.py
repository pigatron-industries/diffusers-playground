from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from enum import Enum


class RotateImageNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 angle:FloatFuncType = 0.0, 
                 fill_colour:ColourFuncType = "black",
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("angle", angle, float)
        self.addParam("fill_colour", fill_colour, str)
        
        
    def process(self, image:Image.Image, angle:float, fill_colour:ColourType) -> Image.Image:
        return image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=fill_colour)