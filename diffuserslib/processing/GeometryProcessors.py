from .ImageProcessor import ImageProcessor, ImageContext
from ..batch import RandomPositionArgument, evaluateArguments

from PIL import ImageDraw
import math
import numpy as np

x1 = 0
y1 = 1
x2 = 2
y2 = 3


class DrawRegularShapeProcessor(ImageProcessor):
    def __init__(self, position=RandomPositionArgument(), size=64, sides=4, rotation=0, fill="black", outline=None):
        self.args = {
            "position": position,
            "size": size,
            "fill": fill,
            "outline": outline,
            "sides": sides,
            "rotation": rotation
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        draw = ImageDraw.Draw(context.image)
        pos = (args["position"][0] + context.offset[0], args["position"][1] + context.offset[1])
        draw.regular_polygon(bounding_circle=(pos, args["size"]), n_sides=args["sides"], rotation=args["rotation"], fill=args["fill"], outline=args["outline"])
        return context


class DrawCheckerboardProcessor(ImageProcessor):
    def __init__(self, size=(64, 64), fill="black", start="black"):
        self.args = {
            "size": size,
            "fill": fill,
            "start": start
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        draw = ImageDraw.Draw(context.image)
        square_width = args["size"][0]
        square_height = args["size"][1]
        for i in range(0, context.size[0], square_width):
            for j in range(0, context.size[1], square_height):
                if(args["start"] == "black"):
                    if (i//square_width + j//square_height) % 2 == 0:
                        draw.rectangle([i+context.offset[0], j+context.offset[1], i+square_width+context.offset[0], j+square_height+context.offset[1]], fill=args["fill"])
                else:
                    if (i//square_width + j//square_height) % 2 == 1:
                        draw.rectangle([i+context.offset[0], j+context.offset[1], i+square_width+context.offset[0], j+square_height+context.offset[1]], fill=args["fill"])
        return context


class DrawGeometricSpiralProcessor(ImageProcessor):
    def __init__(self, startx = 0, starty = 0, endx = None, endy = None, outline="white", fill="black", ratio = 1/1.618033988749895, iterations=7, direction="right", turn="clockwise", 
                 draw = (True, True)):
        self.args = {
            "startx": startx,
            "starty": starty,
            "endx": endx,
            "endy": endy,
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
        startx = args["startx"] + context.offset[0]
        starty = args["starty"] + context.offset[1]
        endx = args["endx"]
        endy = args["endy"]
        if endx is None:
            endx = context.viewport[2]-1
        else:
            endx = endx + context.offset[0]
        if endy is None:
            endy = context.viewport[3]-1
        else:
            endy = endy + context.offset[1]

        if(args["turn"] == "clockwise"):
            directions = ["right", "down", "left", "up"]
        else:
            directions = ["right", "up", "left", "down"]
        direction = args["direction"]
        directionIndex = directions.index(direction)

        draw = ImageDraw.Draw(context.image)
        rect = [startx, starty, endx, endy]
        draw.rectangle(rect, fill=args["fill"], outline=args["outline"])

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


class DrawJuliaSetProcessor(ImageProcessor):
    def __init__(self, c_real, c_imaginary, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, max_iter=255):
        self.args = {
            "c_real": c_real,
            "c_imaginary": c_imaginary,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "max_iter": max_iter
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)

        xstep = (args["xmax"] - args["xmin"]) / (context.image.width - 1)
        ystep = (args["ymax"] - args["ymin"]) / (context.image.height - 1)

        for y in range(context.image.width):
            for x in range(context.image.height):
                z = complex(args["xmin"] + x * xstep, args["ymin"] + y * ystep)
                for i in range(args["max_iter"]):
                    z = z*z + complex(args["c_real"], args["c_imaginary"])
                    if abs(z) > 2.0:
                        break
                if i == args["max_iter"]-1:
                    context.image.putpixel((x, y), (255, 255, 255))
                else:
                    context.image.putpixel((x, y), (0, 0, 0))

        return context