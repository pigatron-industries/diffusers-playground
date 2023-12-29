from .VideoProcessor import VideoProcessor
from typing import Self, List
from PIL import Image


class VideoProcessorPipeline(object):

    def __init__(self):
        self.tasks:List[VideoProcessor] = []

    def addTask(self, task:VideoProcessor) -> Self:
        self.tasks.append(task)
        return self

    def __call__(self, input:List[List[Image.Image]] = []):
        frames = input
        for task in self.tasks:
            frames = [task(frames)]
        return frames