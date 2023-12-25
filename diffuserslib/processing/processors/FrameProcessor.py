from .ImageProcessor import ImageProcessor
from typing import Dict, Any, List, Callable


class FrameProcessor(ImageProcessor):
    def __init__(self, args, interpolation:Callable[[float], float]|None = None):
        self.interpolation = interpolation
        self.frame = 0
        super().__init__(args)

    def initFrames(self, frames):
        self.frames = frames
        self.calcTimings()

    def calcTimings(self):
        if(self.interpolation is not None):
            step_size = 1 / self.frames
            frame_positions = [i*step_size for i in range(self.frames+1)]
            self.frametimings = [self.interpolation(x) for x in frame_positions]
            self.frametimings_diff = [self.frametimings[i+1] - self.frametimings[i] for i in range(len(self.frametimings) - 1)]

    def getFrameTimeDiff(self):
        """ Git time difference between current frame and previous frame, as a fraction of the whole transform """
        if(self.frame > 0):
            return self.frametimings_diff[self.frame-1]
        else:
            return 0
