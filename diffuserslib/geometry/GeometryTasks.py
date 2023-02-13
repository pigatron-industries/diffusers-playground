from .GeometryPipeline import GeometryTask
from .. import Argument, evaluateArguments

import random
from PIL import Image, ImageDraw, ImageOps


class RandomPositionArgument(Argument):
    """ Get a random position in an image """
    def __init__(self, border_width = 32):
        self.border_width = border_width

    def __call__(self, context):
        left = self.border_width
        top = self.border_width
        right = context.size[0] - self.border_width
        bottom = context.size[1] - self.border_width
        return (random.randint(left, right), random.randint(top, bottom))


class DrawRegularShape(GeometryTask):
    def __init__(self, position=RandomPositionArgument(), size=64, sides=4, rotation=0, fill="black"):
        self.args = {
            "position": position,
            "size": size,
            "fill": fill,
            "sides": sides,
            "rotation": rotation
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        draw = ImageDraw.Draw(context.image)
        pos = (args["position"][0] + context.offset[0], args["position"][1] + context.offset[1])
        draw.regular_polygon(bounding_circle=(pos, args["size"]), n_sides=args["sides"], rotation=args["rotation"], fill=args["fill"])
        return context


class DrawCheckerboard(GeometryTask):
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


class Symmetrize(GeometryTask):
    """ horizontal, vertical, or rotation """
    def __init__(self, type="horizontal"):
        self.args = {
            "type": type
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        if(args["type"] == "horizontal"):
            modimage = ImageOps.flip(context.image)
        elif(args["type"] == "vertical"):
            modimage =  ImageOps.mirror(context.image)
        elif(args["type"] == "rotation"):
            modimage =  context.image.rotate(180)
        else:
            return context
        context.image.alpha_composite(modimage, (0, 0))
        return context


class Spiralize(GeometryTask):
    def __init__(self, rotation=180, steps=4, zoom=2):
        self.args = {
            "rotation": rotation,
            "steps": steps,
            "zoom": zoom
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        angle = args["rotation"] / args["steps"]
        zoom = ((args["zoom"]-1) / args["steps"]) + 1
        modimages = []
        modimage = context.image
        for i in range(args["steps"]):
            modimage = modimage.rotate(angle)
            modimage = self.zoomOut(modimage, zoom)
            modimages.append(modimage)
        for modimage in modimages:
            context.image.alpha_composite(modimage, (0, 0))
        return context

    def zoomOut(self, image, ratio):
        modimage = Image.new('RGBA', (int(image.width*ratio), int(image.height*ratio)), (255, 255, 255, 0))
        xpos = (modimage.width - image.width) // 2
        ypos = (modimage.height - image.height) // 2
        modimage.paste(image, (xpos, ypos))
        return modimage.resize((image.width, image.height), resample=Image.BICUBIC)
        