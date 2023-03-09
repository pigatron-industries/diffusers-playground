from .ImageProcessor import ImageProcessor, ImageContext
from ..batch import RandomPositionArgument, evaluateArguments

from PIL import ImageDraw


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
    def __init__(self, startx = 0, starty = 0, endx = None, endy = None, outline="white", fill="black", ratio = 1/1.618033988749895, iterations=7, direction="right"):
        self.args = {
            "startx": startx,
            "starty": starty,
            "endx": endx,
            "endy": endy,
            "fill": fill,
            "outline": outline,
            "ratio": ratio,
            "iterations": iterations,
            "direction": direction
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
        draw = ImageDraw.Draw(context.image)

        rect = [startx, starty, endx, endy]
        draw.rectangle(rect, fill=args["fill"], outline=args["outline"])

        direction = args["direction"]
        for i in range(args["iterations"]):
            rect_one, rect_two = self.splitRectangle(rect, ratio, direction)
            draw.rectangle(rect_one, fill=args["fill"], outline=args["outline"])
            
        return context
    
    
    def splitRectangle(self, rect, ratio, direction):
        if(direction == "right"):
            rect_left_width = (rect[2]-rect[0]) * ratio
            rect_left = [rect[0], rect[1], rect[0]+rect_left_width, rect[3]]
            rect_right = [rect_left[2], rect[1], rect[2], rect[3]]
            return rect_left, rect_right
        if(direction == "left"):
            rect_left_width = (rect[2]-rect[0]) * (1-ratio)
            rect_left = [rect[0], rect[1], rect[0]+rect_left_width, rect[3]]
            rect_right = [rect_left[2], rect[1], rect[2], rect[3]]
            return rect_right, rect_left



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