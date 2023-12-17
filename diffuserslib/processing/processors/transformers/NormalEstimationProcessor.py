from ..ImageProcessor import ImageProcessor
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2


class NormalEstimationProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')

    def __call__(self, context):
        image = self.depth_estimator(context.getViewportImage())['predicted_depth'][0]
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
        context.setViewportImage(Image.fromarray(image))
        return context
