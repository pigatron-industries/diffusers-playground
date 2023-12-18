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
        image = context.getFullImage().copy()
        if(args["type"] == "horizontal"):
            modimage = ImageOps.flip(image)
        elif(args["type"] == "vertical"):
            modimage =  ImageOps.mirror(image)
        elif(args["type"] == "rotation"):
            modimage =  image.rotate(180)
        else:
            return context
        image.alpha_composite(modimage, (0, 0))
        context.setFullImage(image)
        return context