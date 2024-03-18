from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.diffusers.ConditioningInputNode import ConditioningInputType
from PIL import Image


class FramesConditioningInputNode(FunctionalNode):
    def __init__(self, 
                 frames: FramesFuncType,
                 model: StringFuncType,
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("scale", scale, float)
        self.addParam("frames", frames, List[Image.Image])


    def process(self, 
                frames: List[Image.Image],
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = frames, type = model, model = model, condscale = scale)
        return conditioning_input
    