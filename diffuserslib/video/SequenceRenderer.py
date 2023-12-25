from ..processing import ImageProcessorPipeline, ImageContext
from ..batch import evaluateArgument
from .FrameProcessorPipeline import FrameProcessorPipeline

from typing import Tuple, Callable
from IPython.display import display



class SequenceRenderer():

    def __init__(self, initImage:ImageContext|Callable[[], ImageContext]|None, 
                 frameProcessor:FrameProcessorPipeline, frames:int):
        self.initImage = initImage
        self.frameProcessor = frameProcessor
        self.frames = frames


    def __call__(self):
        self.feedforward(self.initImage)
        for frame in range(self.frames+1):
            output = self.frameProcessor()
            feedforward = self.frameProcessor.getFeedForwardImage()
            self.feedforward(feedforward)
            display(output.getViewportImage())


    def feedforward(self, image:ImageContext|Callable[[], ImageContext]|None):
        feedforward = evaluateArgument(image)
        if(feedforward is not None):
            self.frameProcessor.setPlaceholder("feedforward", feedforward)
