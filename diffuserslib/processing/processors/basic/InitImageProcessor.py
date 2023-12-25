from .. import ImageProcessor, ImageContext
from typing import Dict, Any, List
    

class InitImageProcessor(ImageProcessor):
    def __init__(self, image):
        # We add all arguments to an args dictionary, 
        # some of them may be instances of Argument class which decides what the actual argument should be when it's ready to be used
        args = {
            "image": image
        }
        super().__init__(args)

    def setImage(self, image):
        self.args["image"] = image

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = args["image"]
        if(isinstance(image, ImageContext)):
            outputImage.setFullImage(image.getFullImage())
        else:
            outputImage.setViewportImage(image)
        return outputImage
    