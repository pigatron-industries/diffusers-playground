from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image


class NewImageNode(FunctionalNode):
    def __init__(self, 
                 name:str = "new_image",
                 size:SizeFuncType = (512, 512),
                 background_colour: ColourFuncType = "black"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("background_colour", background_colour, ColourType)


    def process(self, size: SizeType, background_colour: ColourType) -> Image.Image:
        return Image.new("RGB", size, background_colour)
    