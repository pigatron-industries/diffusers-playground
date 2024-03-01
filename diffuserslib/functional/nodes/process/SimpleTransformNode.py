from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from PIL import ImageOps, Image
from enum import Enum


class SimpleTransformNode(FunctionalNode):

    class TransformType(Enum):
        NONE = "none"
        FLIPHORIZONTAL = "fliphorizontal"
        FLIPVERTICAL = "flipvertical"
        ROTATE90 = "rotate90"
        ROTATE180 = "rotate180"
        ROTATE270 = "rotate270"


    def __init__(self, 
                 image:ImageFuncType,
                 type:StringFuncType = TransformType.FLIPHORIZONTAL.value, 
                 name:str="simple_transform"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("type", type, str)


    def process(self, image:Image.Image, type:str) -> Image.Image:
        image = image.copy()
        if(type == self.TransformType.FLIPVERTICAL):
            image = ImageOps.flip(image)
        elif(type == self.TransformType.FLIPHORIZONTAL):
            image = ImageOps.mirror(image)
        elif(type == self.TransformType.ROTATE180):
            image = image.transpose(Image.ROTATE_180)
        elif(type == self.TransformType.ROTATE90):
            image = image.transpose(Image.ROTATE_90)
        elif(type == self.TransformType.ROTATE270):
            image = image.transpose(Image.ROTATE_270)
        else:
            image = image
        return image