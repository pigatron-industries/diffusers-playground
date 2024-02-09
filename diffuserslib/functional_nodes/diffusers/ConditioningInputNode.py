from ..FunctionalNode import FunctionalNode, TypeInfo
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
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("image", image, TypeInfo("Image"))
        self.addParam("model", model, TypeInfo("Model.conditioning"))
        self.addParam("scale", scale, TypeInfo("float"))


    def process(self, 
                image: Image.Image,
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, type = model, model = model, condscale = scale)
        return conditioning_input
    