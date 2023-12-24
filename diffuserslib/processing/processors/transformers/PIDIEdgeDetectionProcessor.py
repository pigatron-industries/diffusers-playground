from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux import PidiNetDetector
from typing import Dict, Any, List


class PIDIEdgeDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.pidi(inputImages[0].getViewportImage())
        outputImage.setViewportImage(image)
        return outputImage