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
                 scale: FloatFuncType = 1.0,
                 name:str = "conditioning_input"):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("scale", scale, float)
        self.addParam("image", image, Image.Image)


    def process(self, 
                image: Image.Image,
                model: str,
                scale: float = 1.0) -> ConditioningInputType:
        conditioning_input = ConditioningInputType(image = image, type = model, model = model, condscale = scale)
        return conditioning_input
    