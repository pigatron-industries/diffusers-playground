from diffuserslib.util import DeepCopyObject
from typing import Dict, Any, List, Self, Tuple
from dataclasses import dataclass


@dataclass
class NodeParameter:
    node: str
    name: str
    type: type
    value: Any
    initial_value: Any
    evaluated: Any = None


@dataclass
class WorkflowProgress:
    progress: float
    output: Any


class WorkflowInterruptedException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FunctionalNode(DeepCopyObject):
    name = "FunctionalNode"

    def __init__(self, node_name:str):
        super().__init__()
        self.node_name = node_name
        self.params:Dict[str, NodeParameter] = {}
        self.initparams:Dict[str, NodeParameter] = {}
        self.inited = False
        self.output = None
        self.previous_outputs = []
        self.previous_outputs_size = 1
        self.stopping = False


    def __call__(self) -> Any:
        if(self.output is not None):
            return self.output
        args = self.evaluateParams()
        self.output = self.process(**args)
        return self.output
    

    def flush(self):
        """Flush the node's output. This is called between animation frames to force regeneration in the next frame."""
        self.previous_outputs.append(self.output)
        if(len(self.previous_outputs) > self.previous_outputs_size):
            self.previous_outputs.pop(0)
        self.output = None
        self.stopping = False
        self.recursive_action("flush")
        self.recursive_action("flush", init_params=True)


    def reset(self):
        """Reset the node to its initial state. This is called when the workflow is reset."""
        self.previous_output = None
        self.output = None
        self.stopping = False
        args = self.evaluateInitParams()
        self.init(**args)
        self.recursive_action("reset")
        self.recursive_action("reset", init_params=True)
        self.inited = True


    def stop(self):
        self.stopping = True
        self.recursive_action("stop")


    def getNode(self, node_name:str) -> Self|None:
        if(self.node_name == node_name):
            return self
        return self.recursive_action("getNode", return_value=True, node_name=node_name)


    def getProgress(self) -> WorkflowProgress|None:
        return self.recursive_action("getProgress", return_value=True)


    def recursive_action(self, action:str, return_value:bool=False, init_params:bool=False, **kwargs):
        params = self.initparams if(init_params) else self.params
        for paramname, param in params.items():
            if(isinstance(param.value, FunctionalNode)):
                func = getattr(param.value, action)
                result = func(**kwargs)
                if(return_value and result is not None):
                    return result
            elif(isinstance(param.value, List)):
                for listvalue in param.value:
                    if(isinstance(listvalue, FunctionalNode)):
                        func = getattr(listvalue, action)
                        result = func(**kwargs)
                        if(return_value and result is not None):
                            return result


    def init(self, **kwargs):
        pass


    def process(self, **kwargs):
        raise Exception("Not implemented")
    

    def getPreviousOutput(self, index=-1) -> Any:
        try:
            return self.previous_outputs[index]
        except IndexError:
            return None
        
    
    def addParam(self, name:str, value:Any, type:type):
        self.params[name] = NodeParameter(node=self.node_name, name=name, value=value, initial_value=value, type=type)


    def addInitParam(self, name:str, value:Any, type:type):
        self.initparams[name] = NodeParameter(node=self.node_name, name=name, value=value, initial_value=value, type=type)


    def getParams(self) -> List[NodeParameter]:
        return list(self.params.values())


    def getInitParams(self) -> List[NodeParameter]:
        return list(self.initparams.values())


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
    

    def evaluateInitParams(self):
        paramvalues = {}
        for paramname, param in self.initparams.items():
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
        paramdicts = [self.initparams, self.params]
        for paramdict in paramdicts:
            for paramname, param in paramdict.items():
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
    

    def setValue(self, value:Any):
        pass


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