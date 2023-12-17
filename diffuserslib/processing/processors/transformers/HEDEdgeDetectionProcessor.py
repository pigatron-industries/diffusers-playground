from ..ImageProcessor import ImageProcessor
from controlnet_aux import HEDdetector


class HEDEdgeDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    def __call__(self, context):
        image = self.hed(context.getViewportImage())
        context.setViewportImage(image)
        return context
