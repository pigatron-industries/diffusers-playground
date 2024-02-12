from ...FunctionalNode import FunctionalNode
from ...FunctionalTyping import *
from ....inference.GenerationParameters import ControlImageParameters, ControlImageType
from PIL import Image

ConditioningInputType = ControlImageParameters
ConditioningInputFuncType = ConditioningInputType | Callable[[], ConditioningInputType]
ConditioningInputFuncsType = List[ConditioningInputType] | List[Callable[[], ConditioningInputType]]

class ConditioningInputNode(FunctionalNode):
    def __init__(self, 
                 image: ImageFuncType,
                 model: StringFuncType,
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("model", model, ConditioningInputType)
        self.addParam("scale", scale, float)


    def process(self, 
                image: Image.Image,
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, type = model, model = model, condscale = scale)
        return conditioning_input
    