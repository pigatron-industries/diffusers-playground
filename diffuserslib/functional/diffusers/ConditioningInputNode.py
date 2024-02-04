from ..FunctionalNode import FunctionalNode
from ..FunctionalTyping import *
from ...inference.GenerationParameters import ControlImageParameters, ControlImageType
from PIL import Image

ConditioningInputType = ControlImageParameters
ConditioningInputFuncType = ConditioningInputType | Callable[[], ConditioningInputType]

class ConditioningInputNode(FunctionalNode):
    def __init__(self, 
                 image: ImageFuncType,
                 model: StringFuncType,
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input",):
        args = {
            "image": image,
            "model": model,
            "scale": scale
        }
        super().__init__("conditioning_input", args)


    def process(self, 
                image: Image.Image,
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, type = model, model = model, condscale = scale)
        return conditioning_input
    