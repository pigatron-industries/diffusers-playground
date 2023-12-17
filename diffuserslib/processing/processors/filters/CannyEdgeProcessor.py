from ..ImageProcessor import ImageProcessor
from ....batch import evaluateArguments
from ....ImageUtils import pilToCv2

import numpy as np
import cv2
from PIL import Image
    


class CannyEdgeProcessor(ImageProcessor):
    def __init__(self, low_threshold = 100, high_threshold = 200):
        self.args = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        image = pilToCv2(context.image)
        image = cv2.Canny(image, args["low_threshold"], args["high_threshold"])
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        context.image = Image.fromarray(image)
        return context