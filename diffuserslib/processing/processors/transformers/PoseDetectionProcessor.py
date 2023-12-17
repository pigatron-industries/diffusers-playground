from ..ImageProcessor import ImageProcessor
from controlnet_aux import OpenposeDetector


class PoseDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.pose = OpenposeDetector.from_pretrained('lllyasviel/Annotators')

    def __call__(self, context):
        image = self.pose(context.getViewportImage())
        context.setViewportImage(image)
        return context