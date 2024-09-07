from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class NoOpNode(FunctionalNode):

    def __init__(self, 
                 input:Any, 
                 name:str="resize_image"):
        super().__init__(name)
        self.addParam("input", input, Any)
        
        
    def process(self, input:Any) -> Any:
        return input
