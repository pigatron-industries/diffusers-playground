from ...ProcessingPipeline import ImageProcessor, ImageContext
from ....batch import RandomPositionArgument, evaluateArguments

from PIL import ImageDraw

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

