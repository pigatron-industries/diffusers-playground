from ...FunctionalNode import FunctionalNode, TypeInfo
from ...FunctionalTyping import *
from PIL import ImageDraw, Image


class NewImageNode(FunctionalNode):
    def __init__(self, 
                 name:str = "new_image",
                 size: SizeFuncType = (512, 512),
                 background_colour: ColourFuncType = "black"):
        super().__init__(name)
        self.addParam("size", size, TypeInfo("Size"))
        self.addParam("background_colour", background_colour, TypeInfo("Colour"))


    def process(self, size: SizeType, background_colour: ColourType) -> Image.Image:
        return Image.new("RGB", size, background_colour)
    