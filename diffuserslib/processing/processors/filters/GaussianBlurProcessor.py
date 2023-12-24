from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import ImageFilter
from typing import Dict, Any, List
    

class GaussianBlurProcessor(ImageProcessor):
    def __init__(self, radius = 2):
        args = {
            "radius": radius
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage()
        image = image.filter(ImageFilter.GaussianBlur(args["radius"]))
        outputImage.setFullImage(image)
        return outputImage