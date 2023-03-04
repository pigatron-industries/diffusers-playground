from .ImageProcessor import ImageProcessor
from .. import evaluateArguments
from ..ImageUtils import cv2ToPil, pilToCv2

from PIL import ImageEnhance, ImageFilter
import numpy as np
import cv2
from PIL import Image


class SaturationProcessor(ImageProcessor):
    def __init__(self, saturation = 0):
        self.args = {
            "saturation": saturation
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        converter = ImageEnhance.Color(context.image)
        context.image = converter.enhance(args["saturation"]+1)
        return context
    

class GaussianBlurProcessor(ImageProcessor):
    def __init__(self, radius = 2):
        self.args = {
            "radius": radius
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)
        context.image = context.image.filter(ImageFilter.GaussianBlur(args["radius"]))
        return context
    

class GaussianNoiseProcessor(ImageProcessor):
    def __init__(self, sigma = 2):
        self.args = {
            "sigma": sigma
        }

    def __call__(self, context):
        args = evaluateArguments(self.args, context=context)

        image = pilToCv2(context.image)
        row, col, ch = image.shape
        gauss = np.random.normal(0, args["sigma"], (row, col, ch))
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        context.image = cv2ToPil(noisy_image)
        return context
    

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