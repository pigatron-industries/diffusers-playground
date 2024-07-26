from regex import F
from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from typing import Dict, Callable, Any


class GenericNode(FunctionalNode):
    def __init__(self, function, name:str = "generic", **kwargs):
        super().__init__(name)
        self.addParam("function", function, Callable)
        for key, value in kwargs.items():
            self.addParam(key, value, Any)
        

    def process(self, function:Callable, **kwargs) -> Any:
        return function(**kwargs)
