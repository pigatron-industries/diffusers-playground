from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class FrameSplitterNode(FunctionalNode):
    def __init__(self, 
                 frames:FramesFuncType,
                 name:str = "frame_splitter"):
        super().__init__(name)
        self.addInitParam("frames", frames, List[Image.Image])
        self.framenum = 0
    

    def init(self, frames:List[Image.Image]):
        self.frames = frames
        self.framenum = 0


    def process(self) -> Image.Image|None:
        if(self.framenum < len(self.frames)):
            frame = self.frames[self.framenum]
            self.framenum += 1
            return frame
        else:
            return None
