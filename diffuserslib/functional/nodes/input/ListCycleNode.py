from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class ListCycleNode(FunctionalNode):
    def __init__(self, items:ListFuncType, name:str = "list_cycle"):
        super().__init__(name)
        self.items = []
        self.item_num = 0
        self.addInitParam("items", items, List[Any])


    def init(self, items:List[Any]):
        self.items = items
        self.item_num = 0


    def process(self) -> Any:
        item = self.items[self.item_num]
        self.item_num += 1
        if self.item_num >= len(self.items):
            self.item_num = 0
        return item
