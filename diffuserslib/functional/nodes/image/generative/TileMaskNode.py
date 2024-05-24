from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *

from PIL import Image, ImageDraw


class TileMaskNode(FunctionalNode):
    def __init__(self, 
                 size:SizeFuncType = (1024, 1024),
                 border:IntFuncType = 64,
                 gradient:IntFuncType = 128,
                 name:str = "tile_mask"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("border", border, int)
        self.addParam("gradient", gradient, int)


    def process(self, size:SizeType, border:int, gradient:int) -> Image.Image:
        # TODO different shaped masks
        return TileMaskNode.createMask(size[0], size[1], border, gradient, top=False, bottom=False, left=False, right=False)
        

    @staticmethod
    def createMask(width:int, height:int, border:int, gradient:int, top=False, bottom=False, left=False, right=False):
        mask = Image.new('L', (width, height), color=0xFF)
        draw = ImageDraw.Draw(mask)
        colour = 0
        i = 0
        shape = ((0,0), (width, height))
        while i < border+gradient:
            draw.rectangle(shape, fill = colour)
            i += 1
            if i > border:
                colour = int(256/gradient) * (i-border)
            x1 = 0 if left else i
            y1 = 0 if top else i
            x2 = width if right else width-i
            y2 = height if bottom else height-i
            shape = ((x1, y1), (x2, y2))
        return mask