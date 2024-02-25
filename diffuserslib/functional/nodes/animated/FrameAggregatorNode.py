from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *


class FrameAggregatorNode(FunctionalNode):
    def __init__(self, 
                 frame:ImageFuncType,
                 num_frames:IntFuncType,
                 name:str = "frame_collector"):
        super().__init__(name)
        self.addInitParam("num_frames", num_frames, int)
        self.addParam("frame", frame, Image.Image)
        self.args = {}
        self.frames = []


    def init(self, num_frames:int):
        self.num_frames = num_frames


    def __call__(self) -> List[Image.Image]:
        self.frames = []
        self.frame()
        for i in range(1, self.num_frames):
            self.frame()
        return self.frames
    

    def frame(self) -> Image.Image:
        if(self.stopping):
            raise WorkflowInterruptedException("Workflow interrupted")
        self.flush()
        self.args = self.evaluateParams()
        frame = self.args["frame"]
        self.frames.append(frame)
        return frame
    

    def getProgress(self) -> WorkflowProgress|None:
        if len(self.frames) == 0:
            return WorkflowProgress(0, None)
        else:
            return WorkflowProgress(float(len(self.frames)) / float(self.num_frames), self.frames[-1])
