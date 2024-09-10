

from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from typing import Dict


class GetItemNode(FunctionalNode):
    def __init__(self, 
                 dict:DictFuncType,
                 key:StringFuncType,
                 name:str = "template",
                 **kwargs):
        super().__init__(name)
        self.addParam("dict", dict, Dict[str, Any])
        self.addParam("key", key, str)
        

    def process(self, dict:Dict[str, Any], key:str) -> str:
        if(hasattr(dict, '__dict__')):
            return dict.__dict__[key]
        else:
            return dict[key]
