from ...FunctionalNode import FunctionalNode
from diffuserslib.functional.FunctionalTyping import *
from PIL import ImageDraw, Image
from typing import Tuple


class DrawCheckerboardNode(FunctionalNode):
    def __init__(self, 
                 image:ImageFuncType,
                 blocksize:SizeFuncType = (64, 64), 
                 colour1:ColourFuncType = "black", 
                 colour2:ColourFuncType = "white",
                 name:str = "draw_checkerboard"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("blocksize", blocksize, Tuple[int, int])
        self.addParam("colour1", colour1, str)
        self.addParam("colour2", colour2, str)


    def process(self, image:Image.Image, blocksize:Tuple[int, int], colour1:str, colour2:str) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        square_width = blocksize[0]
        square_height = blocksize[1]
        for i in range(0, image.size[0], square_width):
            for j in range(0, image.size[1], square_height):
                if (i//square_width + j//square_height) % 2 == 0:
                    draw.rectangle((i, j, i+square_width, j+square_height), fill=colour1)
                else:
                    draw.rectangle((i, j, i+square_width, j+square_height), fill=colour2)
        return image

