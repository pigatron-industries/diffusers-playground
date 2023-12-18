from ..processing import ImageProcessorPipeline

from typing import Tuple, Callable


class FrameProcessorPipeline(ImageProcessorPipeline):

    def __init__(self, size:Tuple[int, int]|Callable[[], Tuple[int, int]]|None = None, 
                 oversize:int = 256,
                 feedForwardIndex:int = 0):
        super().__init__(size=size, oversize=oversize)
        self.feedForwardIndex = feedForwardIndex

    def getFeedForwardImage(self):
        # TODO
        pass


class SequenceRenderer():

    def __init__(self, initProcessor:ImageProcessorPipeline, frameProcessor:FrameProcessorPipeline, frames:int):
        self.initProcessor = initProcessor
        self.frameProcessor = frameProcessor
        self.frames = frames


