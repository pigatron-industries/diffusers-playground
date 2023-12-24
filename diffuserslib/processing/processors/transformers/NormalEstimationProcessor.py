from ..ImageProcessor import ImageProcessor, ImageContext
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Any, List


class NormalEstimationProcessor(ImageProcessor):
    def __init__(self):
        self.depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
        super().__init__({})

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        image = self.depth_estimator(inputImages[0].getViewportImage())['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        bg_threhold = 0.4
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        outputImage.setViewportImage(Image.fromarray(image))
        return outputImage
