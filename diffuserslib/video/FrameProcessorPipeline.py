from ..processing import ImageProcessorPipeline, ImageContext

from typing import Tuple, Callable


class FrameProcessorPipeline(ImageProcessorPipeline):

    def __init__(self, size:Tuple[int, int]|Callable[[], Tuple[int, int]]|None = None, 
                 oversize:int = 256,
                 feedForwardIndex:int = 0):
        super().__init__(size=size, oversize=oversize)
        self.feedForwardIndex = feedForwardIndex


    def  __call__(self) -> ImageContext:
        super().__call__()
        return self.getLastOutput()
    

    def getInitImage(self) -> ImageContext:
        return super().getInitImage()
    

    def getFeedForwardImage(self) -> ImageContext:
        return self.tasks[self.feedForwardIndex].getOutputImage()
