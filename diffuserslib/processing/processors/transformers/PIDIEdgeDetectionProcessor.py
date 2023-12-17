from ..ImageProcessor import ImageProcessor
from controlnet_aux import PidiNetDetector


class PIDIEdgeDetectionProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    def __call__(self, context):
        image = self.pidi(context.getViewportImage())
        context.setViewportImage(image)
        return context