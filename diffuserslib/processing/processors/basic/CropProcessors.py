
from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
    
    

class CropProcessor(ImageProcessor):
    def __init__(self, size = (512, 768), position = (0.5, 0.5)):
        self.args = {
            "size": size,
            "position": position
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = context.getViewportImage()
        width = args["size"][0]
        height = args["size"][1]
        left = int((image.width - width) * args["position"][0])
        top = int((image.height - height) * args["position"][1])
        newimage = image.crop((left, top, left+width, top+height))
        context.setViewportImage(newimage)
        return context
