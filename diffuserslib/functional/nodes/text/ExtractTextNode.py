from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
import re


class ExtractTextNode(FunctionalNode):
    def __init__(self, 
                 text:StringFuncType,
                 start_token:StringFuncType,
                 end_token:StringFuncType,
                 name:str = "template"):
        super().__init__(name)
        self.addParam("text", text, str)
        self.addParam("start_token", start_token, str)
        self.addParam("end_token", end_token, str)
        

    def process(self, text:str, start_token:str, end_token:str) -> str:
        pattern = f"{re.escape(start_token)}(.*?){re.escape(end_token)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # print("ExtractTextNode: ", match.group(1))
            return match.group(1)
        else:
            return ""
