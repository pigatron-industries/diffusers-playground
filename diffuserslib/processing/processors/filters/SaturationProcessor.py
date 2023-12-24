from ..ImageProcessor import ImageProcessor, ImageContext
from ....batch import evaluateArguments
from typing import Dict, Any, List
from PIL import ImageEnhance


class SaturationProcessor(ImageProcessor):
    def __init__(self, saturation = 0):
        args = {
            "saturation": saturation
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        converter = ImageEnhance.Color(inputImages[0].getFullImage())
        image = converter.enhance(args["saturation"]+1)
        outputImage.setFullImage(image)
        return outputImage