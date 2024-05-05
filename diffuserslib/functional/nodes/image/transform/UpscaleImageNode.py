from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from enum import Enum
from diffuserslib.imagetools.ESRGAN_upscaler import ESRGANUpscaler


class UpscaleImageNode(FunctionalNode):

    class UpscaleType(Enum):
        ESRGAN = "esrgan"

    UpscaleTypeFuncType = UpscaleType | Callable[[], UpscaleType]

    def __init__(self, 
                 image:ImageFuncType, 
                 type:UpscaleTypeFuncType = UpscaleType.ESRGAN,
                 model:StringFuncType = "4x_remacri",
                 scale:IntFuncType = 4, 
                 name:str="upscale_image"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("type", type, self.UpscaleType)
        self.addParam("model", model, str)
        self.addParam("scale", scale, int)
        
        
    def process(self, image:Image.Image, type:UpscaleType, model:str, scale:int) -> Image.Image|None:
        if(type == self.UpscaleType.ESRGAN):
            return self.upscaleEsrgan(image, scale=scale, model=model)


    def upscaleEsrgan(self, inimage, scale:int, model:str, tilewidth=512+64, tileheight=512+64, overlap=64):
        upscaler = ESRGANUpscaler(f'models/esrgan/{model}.pth', device="mps")
        outimage = upscaler.upscaleTiled(inimage, scale=scale, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap)
        return outimage