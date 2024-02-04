from ..FunctionalNode import FunctionalNode
from ..FunctionalTyping import *
from PIL import ImageDraw, Image


class NewImageNode(FunctionalNode):
    def __init__(self, 
                 size: SizeFuncType = (512, 512),
                 background_colour: ColourFuncType = "black"):
        args = {
            "size": size,
            "background_colour": background_colour
        }
        super().__init__(args)


    def process(self, size: SizeType, background_colour: ColourType) -> Image.Image:
        return Image.new("RGB", size, background_colour)
    