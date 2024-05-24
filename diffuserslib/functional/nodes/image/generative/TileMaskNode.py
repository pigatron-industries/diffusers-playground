from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.ImageUtils import createMask

from PIL import Image


class TileMaskNode(FunctionalNode):
    def __init__(self, 
                 size:SizeFuncType = (1024, 1024),
                 border:IntFuncType = 64,
                 name:str = "tile_mask"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("border", border, int)


    def process(self, size:SizeType, border:int) -> Image.Image:
        # TODO different shaped masks
        return createMask(size[0], size[1], border, top=False, bottom=False, left=False, right=False)
        
    