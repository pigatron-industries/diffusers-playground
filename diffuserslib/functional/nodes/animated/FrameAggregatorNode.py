from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *



class FrameAggregatorNode(FunctionalNode):
    def __init__(self, 
                 frame:ImageFuncType,
                 num_frames:IntFuncType,
                 name:str = "frame_collector"):
        super().__init__(name)
        self.addParam("num_frames", num_frames, int)
        self.addParam("frame", frame, Image.Image)
        self.args = {}
        self.frames = []


    def __call__(self) -> List[Image.Image]:
        self.frames = []
        self.frame()
        num_frames = self.args["num_frames"]
        for i in range(1, num_frames):
            self.frame()
        return self.frames
    

    def frame(self) -> Image.Image:
        self.flush()
        self.args = self.evaluateParams()
        frame = self.args["frame"]
        self.frames.append(frame)
        return frame
    
