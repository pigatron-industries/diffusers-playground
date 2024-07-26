from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from typing import Dict


class TemplateNode(FunctionalNode):
    def __init__(self, 
                 template:StringFuncType,
                 name:str = "template",
                 **kwargs):
        super().__init__(name)
        self.addParam("template", template, str)
        for key, value in kwargs.items():
            self.addParam(key, value, Any)
        

    def process(self, template:str, **kwargs) -> str:
        output = template.format(**kwargs)
        print("TemplateNode:", output)
        return output
