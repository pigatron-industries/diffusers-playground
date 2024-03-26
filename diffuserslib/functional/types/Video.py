from typing import List
from PIL import Image
import tempfile
import io


class Video:
    def __init__(self, frames:List[Image.Image], frame_rate:float, file:tempfile._TemporaryFileWrapper|io.IOBase|None = None):
        self.frames = frames
        self.frame_rate = frame_rate
        self.file = file