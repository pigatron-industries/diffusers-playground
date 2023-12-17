from .ImageContext import ImageContext
from ...batch import evaluateArguments


class ImageProcessor():
    def __init__(self):
        self.args = {}

    def __call__(self, context:ImageContext):
        raise NotImplementedError
    
    def evaluateArguments(self, context:ImageContext):
        return evaluateArguments(self.args, context=context)
