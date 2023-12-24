from .ImageContext import ImageContext
from ...batch import evaluateArguments
from typing import Self, List, Dict, Any
from PIL import Image
from IPython.display import display


class ImageProcessor():
    def __init__(self, args, inputIndexes:List[int]|None = None):
        self.args = args
        if(inputIndexes is None):
            self.inputIndexes = [-1]
        self.inputImages = []
        self.outputImage:ImageContext|None = None

    def __call__(self, inputImages:List[ImageContext])  -> ImageContext:
        outputImage = ImageContext.copy(inputImages[0])
        args = evaluateArguments(self.args, context=outputImage)
        outputImage = self.process(args, inputImages, outputImage)
        self.setOutputImage(outputImage)
        return outputImage
    
    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:
        raise Exception("Not implemented")

    def setInputIndexes(self, inputIndexes:List[int]) -> Self:
        self.inputIndexes = inputIndexes
        return self

    def getInputIndexes(self) -> List[int]:
        return self.inputIndexes
    
    def setInputImages(self, inputImages:List[ImageContext]):
        self.inputImages = inputImages

    def setOutputImage(self, outputImage:ImageContext):
        self.outputImage = outputImage
        # display(self.outputImage.getViewportImage())

    def getOutputImage(self) -> ImageContext:
        if(self.outputImage is not None):
            return self.outputImage
        else:
            raise Exception("No output image")