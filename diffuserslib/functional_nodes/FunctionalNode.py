from typing import Dict, Any, List, Self, Tuple
from dataclasses import dataclass, field
from ..batch import evaluateArguments

@dataclass
class TypeInfo:
    type: str|None = None
    restrict_num: Tuple[float, float, float]|None = None
    restrict_choice: List[Any]|None = None


@dataclass
class ParameterDef:
    node: str
    name: str
    value: Any
    type: TypeInfo


@dataclass
class ParameterInfos:
    params:Dict[str,List[ParameterDef]] = field(default_factory= lambda: {})

    def add(self, node:str, name:str, type:TypeInfo, value:Any):
        if(node not in self.params):
            self.params[node] = []
        self.params[node].append(ParameterDef(node, name, value, type))

    def addAll(self, paramInfos:Self):
        for node in paramInfos.params:
            if(node not in self.params):
                self.params[node] = paramInfos.params[node]


class FunctionalNode:

    def __init__(self, node_name:str):
        self.node_name = node_name
        self.params:Dict[str, ParameterDef] = {}

    def __call__(self) -> Any:
        args = self.evaluateParams()
        return self.process(**args)
    
    def process(self, **kwargs):
        raise Exception("Not implemented")
    
    def addParam(self, name:str, value:Any, type:TypeInfo):
        self.params[name] = ParameterDef(node=self.node_name, name=name, value=value, type=type)
    
    def getInputParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        for paramname in self.params:
            paramdef = self.params[paramname]
            if(isinstance(paramdef.value, FunctionalNode)):
                paramInfos.addAll(paramdef.value.getInputParams())
            elif(isinstance(paramdef.value, List)):
                for i, listvalue in enumerate(paramdef.value):
                    if(isinstance(listvalue, FunctionalNode)):
                        paramInfos.addAll(listvalue.getInputParams())
        return paramInfos


    def getStaticParams(self) -> ParameterInfos:
        paramInfos = ParameterInfos()
        for paramname in self.params:
            paramdef = self.params[paramname]
            if(isinstance(paramdef.value, List) and any(callable(item) for item in paramdef.value)):
                for i, value in enumerate(paramdef.value):
                    if(not callable(value)):
                        paramInfos.add(self.node_name, paramname, paramdef.type, value)
                    elif(isinstance(value, FunctionalNode)):
                        paramInfos.addAll(value.getStaticParams())
            elif(not callable(paramdef.value)):
                paramInfos.add(self.node_name, paramname, paramdef.type, paramdef.value)
            elif(isinstance(paramdef.value, FunctionalNode)):
                paramInfos.addAll(paramdef.value.getStaticParams())
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
    

    def evaluateParams(self):
        outargs = {}
        for paramname in self.params.keys():
            paramdef = self.params[paramname]
            if(callable(paramdef.value)):
                outargs[paramname] = paramdef.value()
            elif(isinstance(paramdef.value, list)):
                outargs[paramname] = []
                for listvalue in paramdef.value:
                    if(callable(listvalue)):
                        outargs[paramname].append(listvalue())
                    else:
                        outargs[paramname].append(listvalue)
            else:
                outargs[paramname] = paramdef.value
        return outargs
    