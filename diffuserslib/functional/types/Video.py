from typing import List
from PIL import Image
import tempfile
import io


class Video:
    def __init__(self, frames:List[Image.Image], file:tempfile._TemporaryFileWrapper|io.IOBase):
        self.frames = frames
        self.file = file