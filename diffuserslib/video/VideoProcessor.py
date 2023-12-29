from typing import List
from PIL import Image

class VideoProcessor():
    def __init__(self):
        pass

    def __call__(self, input:List[List[Image.Image]]) -> List[Image.Image]:
        return input[0]
