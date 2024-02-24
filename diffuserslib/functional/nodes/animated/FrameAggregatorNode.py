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


    def __call__(self) -> List[Image.Image]:
        self.output = []
        args = self.evaluateParams()
        num_frames = args["num_frames"]
        frame = args["frame"]
        self.output.append(frame)
        for i in range(1, num_frames):
            self.flush()
            args = self.evaluateParams()
            frame = args["frame"]
            self.output.append(frame)
        return self.output
    
