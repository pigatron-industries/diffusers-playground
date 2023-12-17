from ..processing import ImageProcessorPipeline

from typing import Tuple, Callable


class FrameProcessorPipeline(ImageProcessorPipeline):

    def __init__(self, size:Tuple[int, int]|Callable[[], Tuple[int, int]]|None = None, 
                 oversize:int = 256,
                 feedForwardIndex:int = 0):
        super().__init__(size=size, oversize=oversize)
        self.feedForwardIndex = feedForwardIndex


class SequenceRenderer():

    def __init__(self, processingPipeline:ImageProcessorPipeline, frames:int):
        self.processingPipeline = processingPipeline
        self.frames = frames


