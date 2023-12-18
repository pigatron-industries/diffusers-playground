from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import Image, ImageOps


class SimpleTransformProcessor(ImageProcessor):
    """ 
        type = none, fliphorizontal, flipvertical, rotate90, rotate180, rotate270 
        rotate90 and rotate270 also swap viewport dimensions
    """
    def __init__(self, type="fliphorizontal"):
        self.args = {
            "type": type
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getFullImage()
        if(args["type"] == "flipvertical"):
            modimage = ImageOps.flip(image)
        elif(args["type"] == "fliphorizontal"):
            modimage = ImageOps.mirror(image)
        elif(args["type"] == "rotate180"):
            modimage = image.transpose(Image.ROTATE_180)
        elif(args["type"] == "rotate90"):
            modimage = image.transpose(Image.ROTATE_90)
        elif(args["type"] == "rotate270"):
            modimage = image.transpose(Image.ROTATE_270)
        else:
            return context
        context.setFullImage(modimage)
        context.calcSize()
        return context