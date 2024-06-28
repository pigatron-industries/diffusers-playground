from typing import List, Callable
from PIL import Image
import tempfile
import io


class Video:
    def __init__(self, frame_rate:float, frames:List[Image.Image]|None = None, file:tempfile._TemporaryFileWrapper|io.IOBase|None = None):
        self.frames = frames
        self.frame_rate = frame_rate
        self.file = file


VideoFuncType = Video|Callable[[], Video]