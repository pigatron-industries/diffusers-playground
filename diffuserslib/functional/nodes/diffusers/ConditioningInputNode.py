from ...FunctionalNode import FunctionalNode
from ...types.FunctionalTyping import *
from ....inference.GenerationParameters import ControlImageParameters, ControlImageType
from PIL import Image

ConditioningInputType = ControlImageParameters
ConditioningInputFuncType = ConditioningInputType | Callable[[], ConditioningInputType]
ConditioningInputFuncsType = List[ConditioningInputType] | List[Callable[[], ConditioningInputType]] | Callable[[], List[ConditioningInputType]]

class ConditioningInputNode(FunctionalNode):
    def __init__(self, 
                 image: ImageFuncType,
                 model: StringFuncType,
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("model", model, ConditioningInputType)
        self.addParam("scale", scale, float)
        self.addParam("image", image, Image.Image)


    def process(self, 
                image: Image.Image,
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, type = model, model = model, condscale = scale)
        return conditioning_input
    