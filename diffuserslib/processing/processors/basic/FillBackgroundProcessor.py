from PIL import Image
from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
    
from IPython.display import display

class FillBackgroundProcessor(ImageProcessor):
    def __init__(self, background="white"):
        self.args = {
            "background": background
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getFullImage().copy()
        background = Image.new("RGBA", size=image.size, color=args["background"])
        background.alpha_composite(image, (0, 0))
        context.setFullImage(background)
        return context
