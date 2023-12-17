from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
    

class InitImageProcessor(ImageProcessor):
    def __init__(self, image):
        # We add all arguments to an args dictionary, 
        # some of them may be instances of Argument class which decides what the actual argument should be when it's ready to be used
        self.args = {
            "image": image
        }

    def setImage(self, image):
        self.args["image"] = image

    def __call__(self, context):
        # evaluateArguments decides which argument values to use at runtime
        args = evaluateArguments(self.args, context=context)
        context.setViewportImage(args["image"])
        return context
    