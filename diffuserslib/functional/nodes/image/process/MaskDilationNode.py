from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image, ImageFilter


class MaskDilationNode(FunctionalNode):
    """ Dilate (grow) the mask by a specified amount and feather (blur) it by specified amount 
        dilation: The amount to dilate the mask by
    """

    def __init__(self, 
                 mask:ImageFuncType,
                 dilation:IntFuncType = 10,
                 feather:IntFuncType = 10,
                 name:str="mask_dilation"):
        super().__init__(name)
        self.addParam("mask", mask, Image.Image)
        self.addParam("dilation", dilation, int)
        self.addParam("feather", feather, bool)
        
        
    def process(self, mask:Image.Image, dilation:int, feather:int) -> Image.Image:
        mask = mask.convert("L")
        mask = mask.filter(ImageFilter.MaxFilter(dilation*2+1))
        if feather:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather//2-1))
        return mask
        
    