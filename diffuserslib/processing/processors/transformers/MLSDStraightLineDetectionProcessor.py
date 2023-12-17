from ..ImageProcessor import ImageProcessor
from controlnet_aux import MLSDdetector


class MLSDStraightLineDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    def __call__(self, context):
        image = self.mlsd(context.getViewportImage())
        context.setViewportImage(image)
        return context