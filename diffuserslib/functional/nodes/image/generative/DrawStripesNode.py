from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import ImageDraw, Image
import math


class DrawStripesNode(FunctionalNode):
    def __init__(self, 
                 image:ImageFuncType,
                 stripewidth:IntFuncType = 64, 
                 stripeangle:FloatFuncType = 45,
                 colour1:ColourFuncType = "black", 
                 colour2:ColourFuncType = "white",
                 name:str = "draw_checkerboard"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("stripewidth", stripewidth, int)
        self.addParam("stripeangle", stripeangle, float)
        self.addParam("colour1", colour1, str)
        self.addParam("colour2", colour2, str)


    def process(self, image:Image.Image, stripewidth:int, stripeangle:float, colour1:str, colour2:str) -> Image.Image:
        stripeimage = Image.new("RGB", (image.width*2, image.height*2), colour1)
        draw = ImageDraw.Draw(stripeimage)

        for i in range(stripewidth//2, stripeimage.width, stripewidth*2):
            draw.rectangle([(i, 0), (i + stripewidth, stripeimage.height)], fill=colour2)

        stripeimage = stripeimage.rotate(stripeangle)

        cropped_image = stripeimage.crop((stripeimage.width//2 - image.width//2, stripeimage.height//2 - image.height//2, stripeimage.width//2 + image.width//2, stripeimage.height//2 + image.height//2))
        image.paste(cropped_image, (0, 0))
        return image

