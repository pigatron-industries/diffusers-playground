from ..ImageProcessor import ImageProcessor, ImageContext
from ....batch import RandomPositionArgument, evaluateArguments

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
