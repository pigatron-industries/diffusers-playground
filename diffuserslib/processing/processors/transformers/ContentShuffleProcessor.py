from ..ImageProcessor import ImageProcessor, ImageContext
from controlnet_aux import ContentShuffleDetector
from typing import Dict, Any, List


class ContentShuffleProcessor(ImageProcessor):
    def __init__(self):
        self.content_shuffle = ContentShuffleDetector()
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.content_shuffle(inputImages[0].getViewportImage())
        outputImage.setViewportImage(image)
        return outputImage
