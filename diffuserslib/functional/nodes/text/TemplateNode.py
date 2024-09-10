from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from typing import Dict


class TemplateNode(FunctionalNode):
    def __init__(self, 
                 template:StringFuncType,
                 dictinput:Dict[str, Any] = {},
                 name:str = "template",
                 **kwargs):
        super().__init__(name)
        self.addParam("template", template, str)
        self.addParam("dictinput", dictinput, Dict[str, Any])
        for key, value in kwargs.items():
            self.addParam(key, value, Any)
        

    def process(self, template:str, dictinput:Dict[str, Any], **kwargs) -> str:
        items = {}
        if(dictinput is not None):
            if(hasattr(dictinput, '__dict__')):
                items.update(dictinput.__dict__)
            else:
                items.update(dictinput)
        items.update(kwargs)
        output = template.format(**items)
        print("TemplateNode:", output)
        return output
