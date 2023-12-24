from ..ImageProcessor import ImageProcessor, ImageContext
from ....ImageUtils import pilToCv2

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List


class CannyEdgeProcessor(ImageProcessor):
    def __init__(self, low_threshold = 100, high_threshold = 200):
        args = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = pilToCv2(inputImages[0].getFullImage())
        image = cv2.Canny(image, args["low_threshold"], args["high_threshold"])
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        outputImage.setFullImage(Image.fromarray(image))
        return outputImage