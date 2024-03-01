from ...FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import ImageDraw, Image
from typing import Tuple
from enum import Enum
import math
import numpy as np

GOLDEN_RATIO = 1/1.618033988749895

x1 = 0
y1 = 1
x2 = 2
y2 = 3

DrawOptionsType = Tuple[bool, bool]
DrawOptionsFuncType = DrawOptionsType | Callable[[], DrawOptionsType]


class DrawGeometricSpiralNode(FunctionalNode):

    class Direction(Enum):
        RIGHT = "right"
        LEFT = "left"
        UP = "up"
        DOWN = "down"

    class Turn(Enum):
        CLOCKWISE = "clockwise"
        ANTICLOCKWISE = "anticlockwise"


    def __init__(self, 
                 image:ImageFuncType,
                 rect:RectFuncType = (0, 0, 1, 1), 
                 outline_colour:ColourFuncType = "white", 
                 fill_colour:ColourFuncType = "black", 
                 ratio:FloatFuncType = GOLDEN_RATIO, 
                 iterations:IntFuncType = 7, 
                 direction:StringFuncType = Direction.RIGHT.value, 
                 turn:StringFuncType = Turn.CLOCKWISE.value,
                 draw_options:DrawOptionsFuncType = (True, True),
                 name:str = "geometric_spiral"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("rect", rect, RectType)
        self.addParam("outline_colour", outline_colour, ColourType)
        self.addParam("fill_colour", fill_colour, ColourType)
        self.addParam("ratio", ratio, float)
        self.addParam("iterations", iterations, int)
        self.addParam("direction", direction, str)
        self.addParam("turn", turn, str)
        self.addParam("draw_options", draw_options, DrawOptionsType)


    def process(self, image:Image.Image, 
                rect:Tuple[int, int, int, int], 
                outline_colour:ColourType, 
                fill_colour:ColourType, 
                ratio:float, 
                iterations:int, 
                direction:str, 
                turn:str, 
                draw_options:DrawOptionsType) -> Image.Image:
        startx = rect[0] * image.size[0]
        starty = rect[1] * image.size[1]
        endx = rect[2] * image.size[0]
        endy = rect[3] * image.size[1]
        rectpix = (startx, starty, endx, endy)

        if(turn == self.Turn.CLOCKWISE.value):
            directions = [self.Direction.RIGHT.value, self.Direction.DOWN.value, self.Direction.LEFT.value, self.Direction.UP.value]
        else:
            directions = [self.Direction.RIGHT.value, self.Direction.UP.value, self.Direction.LEFT.value, self.Direction.DOWN.value]
        directionIndex = directions.index(direction)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for i in range(iterations):
            rect_one, rect_two = self.splitRectangle(rectpix, ratio, direction)
            self.drawSegment(draw, rect_one, direction=direction, turn=turn, fill=fill_colour, outline=outline_colour, drawRectangle=draw_options[0], drawSpiral=draw_options[1])
            rectpix = rect_two
            directionIndex = (directionIndex + 1) % 4
            direction = directions[directionIndex]
            
        return image
    

    def drawSegment(self, draw, rect, direction, turn, fill, outline, drawRectangle, drawSpiral):
        if(drawRectangle):
            draw.rectangle(rect, fill=fill, outline=outline)

        if(drawSpiral):
            width = rect[x2] - rect[x1]
            height = rect[y2] - rect[y1]
            if(direction == "right"):
                arc_centre = (rect[x2], rect[y2])
                arc_height = height * -1
                arc_width = width * -1
            if(direction == "left"):
                arc_centre = (rect[x1], rect[y1])
                arc_height = height * 1
                arc_width = width * 1
            if(direction == "down"):
                arc_centre = (rect[x1], rect[y2])
                arc_height = height * -1
                arc_width = width * 1
            if(direction == "up"):
                arc_centre = (rect[x2], rect[y1])
                arc_height = height * 1
                arc_width = width * -1
            if(turn != "clockwise" and direction in ("up", "down")):
                arc_centre = (arc_centre[0]+arc_width, arc_centre[1])
                arc_width = arc_width * -1
            if(turn != "clockwise" and direction in ("left", "right")):
                arc_centre = (arc_centre[0], arc_centre[1]+arc_height)
                arc_height = arc_height * -1

            x_start = arc_centre[0]+arc_width
            y_start = arc_centre[1]
            stepsize = math.pi/32
            for angle in np.arange(stepsize, math.pi/2+stepsize, stepsize):
                x_end = arc_centre[0] + (arc_width * math.cos(angle))
                y_end = arc_centre[1] + (arc_height * math.sin(angle))
                draw.line((int(x_start), int(y_start), int(x_end), int(y_end)), fill=outline)
                x_start = x_end
                y_start = y_end


    def splitRectangle(self, rect:RectType, ratio:float, direction:str) -> Tuple[RectType, RectType]:
        if(direction == "right"):
            rect_left_width = (rect[x2]-rect[x1]) * ratio
            rect_left = (rect[x1], rect[y1], rect[x1]+rect_left_width, rect[y2])
            rect_right = (rect_left[x2], rect[y1], rect[x2], rect[y2])
            return rect_left, rect_right
        elif(direction == "left"):
            rect_left_width = (rect[x2]-rect[x1]) * (1-ratio)
            rect_left = (rect[x1], rect[y1], rect[x1]+rect_left_width, rect[y2])
            rect_right = (rect_left[x2], rect[y1], rect[x2], rect[y2])
            return rect_right, rect_left
        elif(direction == "down"):
            rect_top_height = (rect[y2]-rect[y1]) * ratio
            rect_top = (rect[x1], rect[y1], rect[x2], rect[y1]+rect_top_height)
            rect_bottom = (rect[x1], rect_top[y2], rect[x2], rect[y2])
            return rect_top, rect_bottom
        else: #direction == "up"
            rect_top_height = (rect[y2]-rect[y1]) * (1-ratio)
            rect_top = (rect[x1], rect[y1], rect[x2], rect[y1]+rect_top_height)
            rect_bottom = (rect[x1], rect_top[y2], rect[x2], rect[y2])
            return rect_bottom, rect_top
