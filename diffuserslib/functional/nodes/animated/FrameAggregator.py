from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *



class FrameAggregator(FunctionalNode):
    def __init__(self, 
                 frame:ImageFuncType,
                 num_frames:int,
                 name:str = "frame_collector"):
        super().__init__(name)
        self.num_frames = num_frames
        self.addParam("frame", frame, Image.Image)


    def __call__(self) -> Any:
        for i in range(self.num_frames):
            self.flush()
            args = self.evaluateParams()
            return self.process(**args)
    

    def process(self, points:Points2DType) -> Points2DType:
        return points 
    