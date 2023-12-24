from ..ImageProcessor import ImageProcessor, ImageContext
from transformers import pipeline
from PIL import Image
import numpy as np
from typing import Dict, Any, List


class DepthEstimationProcessor(ImageProcessor):
    def __init__(self):
        self.depth_estimator = pipeline('depth-estimation', model='Intel/dpt-large')
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.depth_estimator(inputImages[0].getFullImage())['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        outputImage.setFullImage(Image.fromarray(image))
        return outputImage
    