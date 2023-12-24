from ..ImageProcessor import ImageProcessor, ImageContext
from ....ImageUtils import cv2ToPil, pilToCv2
from typing import Dict, Any, List
import numpy as np

    
class GaussianNoiseProcessor(ImageProcessor):
    def __init__(self, sigma = 2):
        args = {
            "sigma": sigma
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = pilToCv2(inputImages[0].getFullImage())
        row, col, ch = image.shape
        gauss = np.random.normal(0, args["sigma"], (row, col, ch))
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        outputImage.setFullImage(cv2ToPil(noisy_image))
        return outputImage
    