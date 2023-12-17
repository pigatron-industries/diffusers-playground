from ..ImageProcessor import ImageProcessor
from transformers import pipeline
from PIL import Image
import numpy as np


class DepthEstimationProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.depth_estimator = pipeline('depth-estimation', model='Intel/dpt-large')

    def __call__(self, context):
        image = self.depth_estimator(context.image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        context.image = Image.fromarray(image)
        return context
    