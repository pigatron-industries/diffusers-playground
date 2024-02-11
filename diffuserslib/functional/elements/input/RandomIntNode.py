from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import random


class RandomIntNode(FunctionalNode):
    """ Select a random number between min and max """
    def __init__(self, min:IntFuncType=0, max:IntFuncType=100, name:str="random_int"):
        super().__init__(name)
        print(self.node_name)
        self.addParam("min", min, TypeInfo(ParamType.INT))
        self.addParam("max", max, TypeInfo(ParamType.INT))
        
        
    def process(self, min:int, max:int) -> int:
        return random.randint(min, max)
