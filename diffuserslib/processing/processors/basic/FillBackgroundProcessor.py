from PIL import Image
from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
    

class FillBackgroundProcessor(ImageProcessor):
    def __init__(self, background="white"):
        self.args = {
            "background": background
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        background = Image.new("RGBA", size=context.image.size, color=args["background"])
        background.alpha_composite(context.image, (0, 0))
        context.image = background
        return context
