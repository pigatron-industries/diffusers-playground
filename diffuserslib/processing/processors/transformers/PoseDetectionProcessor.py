from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux.processor import Processor
from typing import Dict, Any, List


class PoseDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.processor = Processor("openpose_full")
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.processor(inputImages[0].getViewportImage(), to_pil=True)
        outputImage.setViewportImage(image)
        return outputImage