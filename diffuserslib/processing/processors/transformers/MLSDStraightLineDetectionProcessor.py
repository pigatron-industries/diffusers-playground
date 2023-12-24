from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux import MLSDdetector
from typing import Dict, Any, List


class MLSDStraightLineDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.mlsd(inputImages[0].getViewportImage())
        outputImage.setViewportImage(image)
        return outputImage
    