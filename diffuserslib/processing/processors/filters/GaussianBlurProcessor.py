from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import ImageFilter
    

class GaussianBlurProcessor(ImageProcessor):
    def __init__(self, radius = 2):
        self.args = {
            "radius": radius
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getFullImage()
        image = image.filter(ImageFilter.GaussianBlur(args["radius"]))
        context.setFullImage(image)
        return context