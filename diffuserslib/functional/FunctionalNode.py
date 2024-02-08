from typing import Dict, Any, List, Self
from dataclasses import dataclass, field
from ..batch import evaluateArguments


@dataclass
class ParameterInfo:
    node: str
    name: str
    type: type
    value: Any


@dataclass
class ParameterInfos:
    params:Dict[str,List[ParameterInfo]] = field(default_factory= lambda: {})

    def add(self, node:str, name:str, type:type, value:Any):
        if(node not in self.params):
            self.params[node] = []
        self.params[node].append(ParameterInfo(node, name, type, value))

    def addAll(self, paramInfos:Self):
        for node in paramInfos.params:
            if(node not in self.params):
                self.params[node] = paramInfos.params[node]


class FunctionalNode:

    def __init__(self, node_name, args:Dict[str, Any]):
        self.node_name = node_name
        self.args = args

    def __call__(self) -> Any:
        args = evaluateArguments(self.args)
        return self.process(**args)
    
    def process(self, **kwargs):
        raise Exception("Not implemented")
    
    def getInputParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        for argname in self.args:
            argvalue = self.args[argname]
            if(isinstance(argvalue, FunctionalNode)):
                paramInfos.addAll(argvalue.getInputParams())
            elif(isinstance(argvalue, List)):
                for i, listvalue in enumerate(argvalue):
                    if(isinstance(listvalue, FunctionalNode)):
                        paramInfos.addAll(listvalue.getInputParams())
        return paramInfos


    def getStaticParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        for argname in self.args:
            argvalue = self.args[argname]
            if(isinstance(argvalue, List) and any(callable(item) for item in argvalue)):
                for i, value in enumerate(argvalue):
                    if(not callable(value)):
                        paramInfos.add(self.node_name, argname, type(value), value)
                    elif(isinstance(value, FunctionalNode)):
                        paramInfos.addAll(value.getStaticParams())
            elif(not callable(argvalue)):
                paramInfos.add(self.node_name, argname, type(argvalue), argvalue)
            elif(isinstance(argvalue, FunctionalNode)):
                paramInfos.addAll(argvalue.getStaticParams())
        return paramInfos
    
    
    def setParam(self, node_name:str, param_name:str, value:Any):
        if(node_name == self.node_name):
            self.args[param_name] = value
        else:
            for argname in self.args:
                argvalue = self.args[argname]
                if(isinstance(argvalue, List) and any(callable(item) for item in argvalue)):
                    for i, listvalue in enumerate(argvalue):
                        if(isinstance(value, FunctionalNode)):
                            listvalue.setParam(node_name, param_name, value)
                elif(isinstance(argvalue, FunctionalNode)):
                    argvalue.setParam(node_name, param_name, value)
    