from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import ImageOps


class SymmetrizeProcessor(ImageProcessor):
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