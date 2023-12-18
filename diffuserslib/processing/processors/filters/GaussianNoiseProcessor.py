from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
from ....ImageUtils import cv2ToPil, pilToCv2

import numpy as np

    
class GaussianNoiseProcessor(ImageProcessor):
    def __init__(self, sigma = 2):
        self.args = {
            "sigma": sigma
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)

        image = pilToCv2(context.getFullImage())
        row, col, ch = image.shape
        gauss = np.random.normal(0, args["sigma"], (row, col, ch))
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        context.setFullImage(cv2ToPil(noisy_image))
        return context
    