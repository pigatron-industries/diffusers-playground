from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image
import random
import os


class ListRandomNode(FunctionalNode):
    def __init__(self, items:ListFuncType, name:str = "list_cycle"):
        super().__init__(name)
        self.items = []
        self.addInitParam("items", items, List[Any])


    def init(self, items:List[Any]):
        self.items = items


    def process(self) -> Any:
        print(self.items)
        item = random.choice(self.items)
        return item
