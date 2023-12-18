from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments

from PIL import ImageEnhance


class SaturationProcessor(ImageProcessor):
    def __init__(self, saturation = 0):
        self.args = {
            "saturation": saturation
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        converter = ImageEnhance.Color(context.getFullImage())
        image = converter.enhance(args["saturation"]+1)
        context.setFullImage(image)
        return context