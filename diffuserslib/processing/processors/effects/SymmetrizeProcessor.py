from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import ImageOps
from typing import Dict, Any, List


class SymmetrizeProcessor(ImageProcessor):
    """ horizontal, vertical, or rotation """
    def __init__(self, type="horizontal"):
        args = {
            "type": type
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage()
        if(args["type"] == "horizontal"):
            modimage = ImageOps.flip(image)
        elif(args["type"] == "vertical"):
            modimage =  ImageOps.mirror(image)
        elif(args["type"] == "rotation"):
            modimage =  image.rotate(180)
        else:
            modimage = image
        image.alpha_composite(modimage, (0, 0))
        outputImage.setFullImage(image)
        return outputImage