from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import random


class RandomItemNode(FunctionalNode):
    """ Pick a random item from a list """
    def __init__(self, items:ListFuncType, name:str="random_item"):
        super().__init__(name)
        self.addParam("items", items, ListType)
        
        
    def process(self, items:ListType) -> Any:
        return random.choice(items)
