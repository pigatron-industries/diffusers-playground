from typing import Dict, Any, List, Self, Tuple
from .FunctionalTyping import TypeInfo
from dataclasses import dataclass, field


@dataclass
class ParameterDef:
    node: str
    name: str
    type: TypeInfo
    value: Any
    initial_value: Any


@dataclass
class ParameterInfos:
    params:Dict[str,List[ParameterDef]] = field(default_factory= lambda: {})

    def add(self, node:str, name:str, type:TypeInfo, value:Any):
        if(node not in self.params):
            self.params[node] = []
        self.params[node].append(ParameterDef(node, name, type, value, value))

    def addAll(self, paramInfos:Self):
        for node in paramInfos.params:
            if(node not in self.params):
                self.params[node] = paramInfos.params[node]


class FunctionalNode:
    name = "FunctionalNode"

    def __init__(self, node_name:str):
        self.node_name = node_name
        self.params:Dict[str, ParameterDef] = {}

    def __call__(self) -> Any:
        args = self.evaluateParams()
        return self.process(**args)
    
    def process(self, **kwargs):
        raise Exception("Not implemented")
    
    def addParam(self, name:str, value:Any, type:TypeInfo):
        self.params[name] = ParameterDef(node=self.node_name, name=name, value=value, initial_value=value, type=type)

    def getParams(self) -> List[ParameterDef]:
        return list(self.params.values())
    
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
    
    
    def setParam(self, node_param_name:str|Tuple[str,str], value:Any, index=None):
        node_name = node_param_name[0] if(isinstance(node_param_name, Tuple)) else self.node_name
        param_name = node_param_name[1] if(isinstance(node_param_name, Tuple)) else node_param_name
        if(node_name == self.node_name):
            if(index is None):
                self.params[param_name].value = value
            else:
                old_value = self.params[param_name].value
                new_value = old_value[:index] + (value,) + old_value[index+1:]
                self.params[param_name].value = new_value
        else:
            for paramname in self.params:
                paramvalue = self.params[paramname].value
                if(isinstance(paramvalue, List) and any(callable(item) for item in paramvalue)):
                    for i, listvalue in enumerate(paramvalue):
                        if(isinstance(listvalue, FunctionalNode)):
                            listvalue.setParam((node_name, param_name), value, index)
                elif(isinstance(paramvalue, FunctionalNode)):
                    paramvalue.setParam((node_name, param_name), value, index)


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
    

    def printDebug(self, level=0):
        print((" "*level*3) + f"* Node: {self.node_name}")
        for paramname in self.params:
            paramdef = self.params[paramname]
            print((" "*level*3) + f"   - {paramname}: {paramdef.value}")
        for paramname in self.params:
            paramdef = self.params[paramname]
            if(isinstance(paramdef.value, List)):
                for i, listvalue in enumerate(paramdef.value):
                    if(isinstance(listvalue, FunctionalNode)):
                        listvalue.printDebug(level+1)
            if(isinstance(paramdef.value, FunctionalNode)):
                paramdef.value.printDebug(level+1)
        return