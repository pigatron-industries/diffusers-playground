from ...FunctionalNode import FunctionalNode, ParameterInfos, TypeInfo
from typing import Any

class InputValueNode(FunctionalNode):
    def __init__(self, name:str="static", type:TypeInfo=TypeInfo(), value:Any=None, mandatory:bool=True):
        super().__init__(name)
        self.mandatory = mandatory
        self.type = type
        self.addParam("value", value, type)

    def setValue(self, value:Any):
        self.params["value"].value = value

    def process(self, value:Any) -> Any:
        if(value is None and self.mandatory):
            raise Exception(f"{self.node_name} is mandatory and has not been set")
        return value
    
    def getInputParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        paramInfos.add(self.node_name, "value", self.type, self.params["value"].value)
        return paramInfos
    
    def getStaticParams(self) -> ParameterInfos:
        return self.getInputParams()