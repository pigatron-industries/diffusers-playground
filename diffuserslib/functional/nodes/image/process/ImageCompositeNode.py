from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image, ImageFilter


class ImageCompositeNode(FunctionalNode):

    def __init__(self, 
                 foreground:ImageFuncType, 
                 background:ImageFuncType,
                 mask:ImageFuncType,
                 name:str="alpha_to_mask"):
        super().__init__(name)
        self.addParam("foreground", foreground, Image.Image)
        self.addParam("background", background, Image.Image)
        self.addParam("mask", mask, Image.Image)
        
        
    def process(self, foreground:Image.Image, background:Image.Image, mask:Image.Image) -> Image.Image:
        foreground = foreground.convert("RGBA")
        background = background.convert("RGBA")
        mask = mask.convert("L")
        return Image.composite(foreground, background, mask)
        
    