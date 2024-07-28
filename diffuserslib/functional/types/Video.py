from typing import List, Callable
from PIL import Image
import tempfile
import io
import cv2


class Video:
    def __init__(self, frame_rate:float|None = None, frames:List[Image.Image]|None = None, file:tempfile._TemporaryFileWrapper|io.IOBase|None = None):
        self.capture = None
        self.frames = frames
        self.frame_rate = frame_rate
        self.file = file
        if(self.frames is not None):
            self.framecount = len(self.frames)
        elif(self.file is not None):
            self._setFrameCountFromFile()
        if(frame_rate is None):
            self._setFrameRateFromFile()


    def getFilename(self) -> str|None:
        if isinstance(self.file, tempfile._TemporaryFileWrapper):
            return self.file.name
        return None
    

    def _setFrameCountFromFile(self):
        if(self.capture is None):
            self.capture = cv2.VideoCapture(self.file.name)
        self.framecount = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    

    def _setFrameRateFromFile(self):
        if(self.capture is None):
            self.capture = cv2.VideoCapture(self.file.name)
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
    

    def getFrameCount(self) -> int:
        return self.framecount
    

    def getFrameRate(self) -> float:
        return self.frame_rate


    def getFrame(self, frame_num:int) -> Image.Image:
        if(self.frames is not None):
            return self.frames[frame_num]
        if(self.capture is None):
            self.capture = cv2.VideoCapture(self.file.name)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.capture.read()
        if not ret:
            raise ValueError("Could not read video file")
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame


VideoFuncType = Video|Callable[[], Video]