from ..ImageProcessor import ImageProcessor
from controlnet_aux import ContentShuffleDetector


class ContentShuffleProcessor(ImageProcessor):
    def __init__(self):
        self.args = {}
        self.content_shuffle = ContentShuffleDetector()

    def __call__(self, context):
        image = self.content_shuffle(context.getViewportImage())
        context.setViewportImage(image)
        return context