from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib import pilToCv2
from controlnet_aux import MLSDdetector


class MLSDStraightLineDetectionNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 name:str="mlsd_staight_line"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        
        
    def process(self, image:Image.Image) -> Image.Image:
        outimage = self.mlsd(image)
        return outimage
    