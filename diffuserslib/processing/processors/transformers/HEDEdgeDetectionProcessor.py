from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux import HEDdetector
from typing import Dict, Any, List


class HEDEdgeDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.hed(inputImages[0].getViewportImage())
        outputImage.setViewportImage(image)
        return outputImage
