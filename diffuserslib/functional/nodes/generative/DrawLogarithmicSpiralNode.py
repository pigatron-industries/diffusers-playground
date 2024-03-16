from ...FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import ImageDraw, Image
from typing import Tuple
from enum import Enum
import math
import numpy as np

x1 = 0
y1 = 1
x2 = 2
y2 = 3


class DrawLogarithmicSpiralNode(FunctionalNode):

    def __init__(self, 
                 image:ImageFuncType,
                 outline_colour:ColourFuncType = "white",
                 revolutions:FloatFuncType = 8,
                 segment_angle:IntFuncType = 1,
                 tightness:FloatFuncType = 0.25,
                 scale:FloatFuncType = 1.0,
                 rotate:IntFuncType = 0,
                 name:str = "logarithmic_spiral"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("outline_colour", outline_colour, ColourType)
        self.addParam("revolutions", revolutions, float)
        self.addParam("segment_angle", segment_angle, int)
        self.addParam("tightness", tightness, float)
        self.addParam("scale", scale, float)
        self.addParam("rotate", rotate, int)


    def process(self, image:Image.Image, 
                outline_colour:ColourType, 
                revolutions:float,
                segment_angle:int,
                tightness:float,
                scale:float,
                rotate:int) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)

        centre_x = image.width // 2
        centre_y = image.height // 2
        rotate_theta = math.radians(rotate)
        x, y = centre_x, centre_y

        for angle in range(0, int(revolutions*360), segment_angle):
            theta = math.radians(angle)
            # TODO make scale an x, y tuple
            radius_new = scale * math.exp(tightness * theta)
            x_new = centre_x + radius_new * math.cos(theta + rotate_theta)
            y_new = centre_y + radius_new * math.sin(theta + rotate_theta)
            draw.line((x, y, x_new, y_new), fill=outline_colour, width=1)
            x, y = x_new, y_new
            
        return image
