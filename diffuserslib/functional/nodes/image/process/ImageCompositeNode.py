from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image, ImageFilter


class ImageCompositeNode(FunctionalNode):

    def __init__(self, 
                 foreground:ImageFuncType, 
                 background:ImageFuncType,
                 mask:ImageFuncType,
                 maskDilation:IntFuncType = 21,
                 maskFeather:IntFuncType = 3,
                 name:str="alpha_to_mask"):
        super().__init__(name)
        self.addParam("foreground", foreground, Image.Image)
        self.addParam("background", background, Image.Image)
        self.addParam("mask", mask, Image.Image)
        self.addParam("maskDilation", maskDilation, int)
        self.addParam("maskFeather", maskFeather, int)
        
        
    def process(self, foreground:Image.Image, background:Image.Image, mask:Image.Image, maskDilation:int, maskFeather:int) -> Image.Image:
        foreground = foreground.convert("RGBA")
        background = background.convert("RGBA")
        mask = mask.convert("L")
        dilated_mask = mask.filter(ImageFilter.MaxFilter(maskDilation))
        feathered_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=maskFeather))
        return Image.composite(foreground, background, feathered_mask)
        
    