from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import ImageDraw
from IPython.display import display


class DrawCheckerboardProcessor(ImageProcessor):
    def __init__(self, size=(64, 64), fill="black", start="black"):
        self.args = {
            "size": size,
            "fill": fill,
            "start": start
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getViewportImage()
        draw = ImageDraw.Draw(image)
        square_width = args["size"][0]
        square_height = args["size"][1]
        for i in range(0, context.size[0], square_width):
            for j in range(0, context.size[1], square_height):
                if(args["start"] == "black"):
                    if (i//square_width + j//square_height) % 2 == 0:
                        draw.rectangle([i, j, i+square_width, j+square_height], fill=args["fill"])
                else:
                    if (i//square_width + j//square_height) % 2 == 1:
                        draw.rectangle([i, j, i+square_width, j+square_height], fill=args["fill"])
        context.setViewportImage(image)
        return context

