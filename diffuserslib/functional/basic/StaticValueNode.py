from ..FunctionalNode import FunctionalNode
from typing import Any

class StaticValueNode(FunctionalNode):
    def __init__(self, name:str|None=None, value:Any=None, mandatory:bool=True):
        self.name = name
        self.value = value
        self.mandatory = mandatory
        super().__init__({})

    def setValue(self, value):
        self.value = value

    def process(self):
        if(self.value is None and self.mandatory):
            raise Exception(f"{self.name} is mandatory and has not been set")
        return self.value