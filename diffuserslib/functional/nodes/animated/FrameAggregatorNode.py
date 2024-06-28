from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from .VideoFrameSplitterNode import VideoFrameSplitterNode


class FrameAggregatorNode(FunctionalNode):
    def __init__(self, 
                 frame:ImageFuncType,
                 num_frames:IntFuncType|None = None,
                 frame_splitter:VideoFrameSplitterNode|None = None,
                 name:str = "frame_aggregator"):
        super().__init__(name)
        self.addParam("frame", frame, Image.Image)
        self.addInitParam("num_frames", num_frames, int)
        self.args = {}
        self.frames = []
        self.frame_splitter = frame_splitter


    def init(self, num_frames:int|None):
        if self.frame_splitter is not None:
            self.num_frames = self.frame_splitter.framecount
        elif num_frames is not None:
            self.num_frames = num_frames
        else:
            raise Exception("num_frames not provided")


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
            return WorkflowProgress(float(len(self.frames)) / float(self.num_frames), self.frames)
