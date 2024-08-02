from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from enum import Enum
from diffuserslib.imagetools.ESRGAN_upscaler import ESRGANUpscaler
from diffuserslib.GlobalConfig import GlobalConfig


class UpscaleImageNode(FunctionalNode):

    class UpscaleType(Enum):
        ESRGAN = "esrgan"
        AURASR = "aurasr"

    UPSCALE_MODELS = {
        "ESRGAN / 4x_remacri": {
            "type": UpscaleType.ESRGAN,
            "model": "4x_remacri",
        },
        "AuraSR": {
            "type": UpscaleType.AURASR,
            "model": None,
        }
    }

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
        self.model = None
        self.modelscale = None
        self.loadedmodel = None
        
        
    def process(self, image:Image.Image, type:UpscaleType, model:str, scale:int) -> Image.Image|None:
        print(type)
        print(model)
        if(type == self.UpscaleType.ESRGAN):
            outimage = self.upscaleEsrgan(image, scale=scale, model=model)
        elif(type == self.UpscaleType.AURASR):
            outimage = self.upscaleAuraSR(image, scale=scale)

        modelscaled = outimage.width / image.width
        if (modelscaled != scale):
            outimage = outimage.resize((image.width*scale, image.height*scale))
        return outimage


    def upscaleEsrgan(self, inimage, scale:int, model:str, tilewidth=512+64, tileheight=512+64, overlap=64):
        upscaler = ESRGANUpscaler(f'models/esrgan/{model}.pth', device="mps")
        outimage = upscaler.upscaleTiled(inimage, scale=scale, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap)
        return outimage
    

    def upscaleAuraSR(self, inimage, scale:int):
        from aura_sr import AuraSR
        if(self.model is None or self.loadedmodel != self.UpscaleType.AURASR):
            self.loadedmodel = self.UpscaleType.AURASR
            self.model = AuraSR.from_pretrained(model_id = "fal/AuraSR-v2")
            self.model.upsampler = self.model.upsampler.to(GlobalConfig.device)
        outimage = self.model.upscale_4x_overlapped(inimage)
        return outimage