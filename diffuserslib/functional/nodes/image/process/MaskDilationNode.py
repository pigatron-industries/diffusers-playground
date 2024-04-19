from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image, ImageFilter


class MaskDilationNode(FunctionalNode):
    """ Dilate (grow) the mask by a specified amount and feather (blur) it by specified amount """

    def __init__(self, 
                 mask:ImageFuncType,
                 dilation:IntFuncType = 21,
                 feather:IntFuncType = 3,
                 name:str="mask_dilation"):
        super().__init__(name)
        self.addParam("mask", mask, Image.Image)
        self.addParam("dilation", dilation, int)
        self.addParam("feather", feather, int)
        
        
    def process(self, mask:Image.Image, dilation:int, feather:int) -> Image.Image:
        mask = mask.convert("L")
        dilated_mask = mask.filter(ImageFilter.MaxFilter(dilation))
        feathered_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=feather))
        return feathered_mask
        
    