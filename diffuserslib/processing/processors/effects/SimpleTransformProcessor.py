from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import Image, ImageOps
from typing import Dict, Any, List


class SimpleTransformProcessor(ImageProcessor):
    """ 
        type = none, fliphorizontal, flipvertical, rotate90, rotate180, rotate270 
        rotate90 and rotate270 also swap viewport dimensions
    """
    def __init__(self, type="fliphorizontal"):
        args = {
            "type": type
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = inputImages[0].getFullImage()
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
            modimage = image
        outputImage.setFullImage(modimage)
        outputImage.calcSize()
        return outputImage