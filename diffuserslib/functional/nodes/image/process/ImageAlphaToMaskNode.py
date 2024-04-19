from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib import pilToCv2
import cv2
import numpy as np


class ImageAlphaToMaskNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType, 
                 smooth:BoolFuncType = False,
                 name:str="alpha_to_mask"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("smooth", smooth, bool)
        
        
    def process(self, image:Image.Image, smooth:bool) -> Image.Image:
        maskimage = Image.new(image.mode, (image.width, image.height), color=(0, 0, 0))
        for x in range(image.width):
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                a = pixel[3]
                if (smooth):
                    maskimage.putpixel((x, y), (255-a, 255-a, 255-a))
                else:
                    if (a < 255):
                        maskimage.putpixel((x, y), (255, 255, 255))
        return maskimage
        
    