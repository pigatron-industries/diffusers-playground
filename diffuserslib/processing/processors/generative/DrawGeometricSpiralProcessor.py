from ..ImageProcessor import ImageProcessor, ImageContext
from ....batch import evaluateArguments

from PIL import ImageDraw
import math
import numpy as np

GOLDEN_RATIO = 1/1.618033988749895

x1 = 0
y1 = 1
x2 = 2
y2 = 3


class DrawGeometricSpiralProcessor(ImageProcessor):
    def __init__(self, rect = (0, 0, 1, 1), outline="white", fill="black", ratio = GOLDEN_RATIO, iterations=7, direction="right", turn="clockwise", 
                 draw = (True, True)):
        self.args = {
            "rect": rect,
            "fill": fill,
            "outline": outline,
            "ratio": ratio,
            "iterations": iterations,
            "direction": direction,
            "turn": turn,
            "draw": draw
        }

    def __call__(self, context:ImageContext):
        args = evaluateArguments(self.args, context=context)
        ratio = args["ratio"]
        inrect = args["rect"]
        startx = inrect[0] * context.size[0] + context.offset[0]
        starty = inrect[1] * context.size[1] + context.offset[1]
        endx = inrect[2] * context.size[0] + context.offset[0]
        endy = inrect[3] * context.size[1] + context.offset[1]
        rect = [startx, starty, endx, endy]

        if(args["turn"] == "clockwise"):
            directions = ["right", "down", "left", "up"]
        else:
            directions = ["right", "up", "left", "down"]
        direction = args["direction"]
        directionIndex = directions.index(direction)

        draw = ImageDraw.Draw(context.image)

        for i in range(args["iterations"]):
            rect_one, rect_two = self.splitRectangle(rect, ratio, direction)
            self.drawSegment(draw, rect_one, direction=direction, turn=args["turn"], fill=args["fill"], outline=args["outline"], drawRectangle=args["draw"][0], drawSpiral=args["draw"][1])
            rect = rect_two
            directionIndex = (directionIndex + 1) % 4
            direction = directions[directionIndex]
            
        return context
    

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
    

    def splitRectangle(self, rect, ratio, direction):
        if(direction == "right"):
            rect_left_width = (rect[x2]-rect[x1]) * ratio
            rect_left = [rect[x1], rect[y1], rect[x1]+rect_left_width, rect[y2]]
            rect_right = [rect_left[x2], rect[y1], rect[x2], rect[y2]]
            return rect_left, rect_right
        if(direction == "left"):
            rect_left_width = (rect[x2]-rect[x1]) * (1-ratio)
            rect_left = [rect[x1], rect[y1], rect[x1]+rect_left_width, rect[y2]]
            rect_right = [rect_left[x2], rect[y1], rect[x2], rect[y2]]
            return rect_right, rect_left
        if(direction == "down"):
            rect_top_height = (rect[y2]-rect[y1]) * ratio
            rect_top = [rect[x1], rect[y1], rect[x2], rect[y1]+rect_top_height]
            rect_bottom = [rect[x1], rect_top[y2], rect[x2], rect[y2]]
            return rect_top, rect_bottom
        if(direction == "up"):
            rect_top_height = (rect[y2]-rect[y1]) * (1-ratio)
            rect_top = [rect[x1], rect[y1], rect[x2], rect[y1]+rect_top_height]
            rect_bottom = [rect[x1], rect_top[y2], rect[x2], rect[y2]]
            return rect_bottom, rect_top
