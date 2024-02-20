from diffuserslib.util import DeepCopyObject
from typing import Dict, Any, List, Self, Tuple, Callable
from dataclasses import dataclass, field
from PIL import Image
import copy


@dataclass
class NodeParameter:
    node: str
    name: str
    type: type
    value: Any
    initial_value: Any
    evaluated: Any = None


@dataclass
class ParameterInfos:
    params:Dict[str,List[NodeParameter]] = field(default_factory= lambda: {})

    def add(self, node:str, name:str, type:type, value:Any):
        if(node not in self.params):
            self.params[node] = []
        self.params[node].append(NodeParameter(node, name, type, value, value))

    def addAll(self, paramInfos:Self):
        for node in paramInfos.params:
            if(node not in self.params):
                self.params[node] = paramInfos.params[node]


class FunctionalNode(DeepCopyObject):
    name = "FunctionalNode"

    def __init__(self, node_name:str):
        super().__init__()
        self.node_name = node_name
        self.params:Dict[str, NodeParameter] = {}

    def __call__(self) -> Any:
        args = self.evaluateParams()
        return self.process(**args)
    

    def update(self):
        # TODO: update should only be called on a node once, and then the node should be marked as updated
        """ Send update signal to all child nodes in case of continuously changing parameters"""
        for paramname, param in self.params.items():
            if(isinstance(param.value, FunctionalNode)):
                param.value.update()
            elif(isinstance(param.value, List)):
                for listvalue in param.value:
                    if(isinstance(listvalue, FunctionalNode)):
                        listvalue.update()

    
    def process(self, **kwargs):
        raise Exception("Not implemented")
    
    def addParam(self, name:str, value:Any, type:type):
        self.params[name] = NodeParameter(node=self.node_name, name=name, value=value, initial_value=value, type=type)

    def getParams(self) -> List[NodeParameter]:
        return list(self.params.values())

    
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
        paramvalues = {}
        for paramname, param in self.params.items():
            if(callable(param.value)):
                paramvalues[paramname] = param.value()
            elif(isinstance(param.value, list)):
                paramvalues[paramname] = []
                for listvalue in param.value:
                    if(callable(listvalue)):
                        paramvalues[paramname].append(listvalue())
                    else:
                        paramvalues[paramname].append(listvalue)
            else:
                paramvalues[paramname] = param.value
            param.evaluated = paramvalues[paramname]
        return paramvalues
    

    def getEvaluatedParamValues(self) -> Dict[str,Dict[str, Any]]:
        nodes = self.getNodes()
        paramnodevalues = {}
        for node in nodes:
            paramvalues = {}
            for paramname, param in node.params.items():
                if(isinstance(param.initial_value, FunctionalNode)): # ignore "hard coded" values
                    paramvalues[paramname] = param.evaluated
            if(paramvalues):
                paramnodevalues[node.node_name] = paramvalues
        return paramnodevalues
    

    def getNodes(self) -> List[Self]:
        nodes = [self]
        for param in self.params.values():
            if(isinstance(param.value, List)):
                for listvalue in param.value:
                    if(isinstance(listvalue, FunctionalNode)):
                        nodes.extend(listvalue.getNodes())
            if(isinstance(param.value, FunctionalNode)):
                nodes.extend(param.value.getNodes())
        return nodes
    

    def visitNodes(self, visitor, parents=[]):
        visitor(self, parents)
        for paramname, param in self.params.items():
            if(isinstance(param.value, FunctionalNode)):
                param.value.visitNodes(visitor, [param]+parents)
            elif(isinstance(param.initial_value, FunctionalNode)):
                param.initial_value.visitNodes(visitor, [param]+parents)
    

    def visitParams(self, visitor, parents=[]):
        output = {}
        for paramname, param in self.params.items():
            output[paramname] = {}
            visitor_output = visitor(param, parents+[param])
            if(visitor_output is not None):
                output[paramname].update(visitor_output)
            if(isinstance(param.value, FunctionalNode)):
                output[paramname].update(param.value.visitParams(visitor, parents+[param]))
            elif(isinstance(param.initial_value, FunctionalNode)):
                output[paramname].update(param.initial_value.visitParams(visitor, parents+[param]))
            elif(isinstance(param.value, List)):
                for i, listvalue in enumerate(param.value):
                    if(isinstance(listvalue, FunctionalNode)):
                        output[paramname][i] = listvalue.visitParams(visitor, parents+[param]+[i])
            if(not output[paramname]):
                del output[paramname]
        return output


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