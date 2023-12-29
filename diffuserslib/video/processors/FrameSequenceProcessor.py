from ...processing import ImageProcessorPipeline, ImageContext
from ...processing.processors import FrameProcessor
from ...batch import evaluateArgument
from ..VideoProcessor import VideoProcessor

from typing import Callable, List
from PIL import Image

from IPython.display import display



class FrameSequenceProcessor(VideoProcessor):

    def __init__(self, initImage:ImageContext|Callable[[], ImageContext]|None, 
                 frameProcessor:ImageProcessorPipeline, frames:int, 
                 feedForwardIndex:int = 0):
        self.initImage = initImage
        self.frameProcessor = frameProcessor
        self.frames = frames
        self.feedForwardIndex = feedForwardIndex


    def __call__(self, input:List[List[Image.Image]]|None = None):
        self.outputFrames = []
        self.initFrames(self.frames)
        self.feedforward(self.initImage)
        for frame in range(self.frames+1):
            output = self.frameProcessor()
            feedforward = self.getFeedForwardImage()
            self.feedforward(feedforward)
            self.outputFrames.append(output.getViewportImage())
            display(output.getViewportImage())
        return self.outputFrames


    def initFrames(self, frames):
        for task in self.frameProcessor.tasks:
            if(isinstance(task, FrameProcessor)):
                task.initFrames(frames)


    def feedforward(self, image:ImageContext|Callable[[], ImageContext]|None):
        feedforward = evaluateArgument(image)
        if(feedforward is not None):
            self.frameProcessor.setPlaceholder("feedforward", feedforward)


    def getFeedForwardImage(self) -> ImageContext:
        return self.frameProcessor.tasks[self.feedForwardIndex].getOutputImage()