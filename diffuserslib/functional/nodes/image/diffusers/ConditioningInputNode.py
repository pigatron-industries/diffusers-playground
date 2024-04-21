from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.GenerationParameters import ControlImageParameters
from PIL import Image

ConditioningInputType = ControlImageParameters
ConditioningInputFuncType = ConditioningInputType | Callable[[], ConditioningInputType]
ConditioningInputFuncsType = List[ConditioningInputType] | List[Callable[[], ConditioningInputType]] | Callable[[], List[ConditioningInputType]]

class ConditioningInputNode(FunctionalNode):
    def __init__(self, 
                 image: ImageFuncType,
                 model: StringFuncType,
                 type: StringFuncType|None = None,
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("scale", scale, float)
        self.addParam("image", image, Image.Image)
        if type is None:
            type = model
        self.addParam("type", type, str)


    def process(self, 
                image: Image.Image,
                model: str,
                type: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, model = model, type = type, condscale = scale)
        return conditioning_input
    