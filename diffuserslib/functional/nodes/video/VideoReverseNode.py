from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types.Video import *


class VideoReverseNode(FunctionalNode):
    def __init__(self, 
                 video:VideoFuncType,
                 name:str = "video_reverse"):
        super().__init__(name)
        self.addParam("video", video, Video)
        self.frames = []


    def process(self, video:Video) -> List[Image.Image]:
        framecount = video.getFrameCount()
        self.frames = []
        for i in range(framecount - 1):
            video.getFrame(i)
            self.frames.insert(0, video.getFrame(i))
            
        return self.frames
    
