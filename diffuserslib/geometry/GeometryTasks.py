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
