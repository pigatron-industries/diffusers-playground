from ..FunctionalNode import FunctionalNode, ParameterInfos
from typing import Any

class StaticValueNode(FunctionalNode):
    def __init__(self, name:str="static", type:type=Any, value:Any=None, mandatory:bool=True):
        self.mandatory = mandatory
        self.type = type
        args = {
            "value": value
        }
        super().__init__(name, args)

    def setValue(self, value:Any):
        self.args["value"] = value

    def process(self, value:Any) -> Any:
        if(value is None and self.mandatory):
            raise Exception(f"{self.node_name} is mandatory and has not been set")
        return value
    
    def getStaticParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        paramInfos.add(self.node_name, "value", self.type, self.args["value"])
        return paramInfos