from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.WorkflowRunner import WorkflowRunner
from .config import *
from typing import List
from dataclasses import dataclass
import inspect
import copy
import importlib
import importlib.util
import sys
import os
import glob

@dataclass
class Model:
    run_type:int = 1
    batch_size:int = 1
    workflow_name:str|None = None
    
    

def str_to_class(str):
    return getattr(sys.modules[__name__], str)
    

class Controller:

    model = Model()
    workflows = []
    workflow:FunctionalNode|None = None
    workflowrunner = WorkflowRunner()

    def __init__(self):
        self.workflows = self.loadWorkflows()
        if(self.model.workflow_name is not None):
            self.loadWorkflow(self.model.workflow_name)
        

    def loadWorkflows(self):
        print("Loading workflows")
        path = os.path.join(os.path.dirname(__file__), '../functional_workflows')
        files = glob.glob(path + '/*.py')
        workflows = {}
        for file in files:
            spec = importlib.util.spec_from_file_location("module.workflow", file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module.workflow"] = module
            spec.loader.exec_module(module)
            workflows[module.name()] = module
        return workflows
    

    def loadWorkflow(self, workflow_name):
        if(self.workflow is not None and self.workflow.name == workflow_name):
            return
        if workflow_name in self.workflows:
            self.model.workflow_name = workflow_name
            self.workflow = self.workflows[workflow_name].build()
            print(f"Loaded workflow: {self.workflow.name}")
            self.workflow.printDebug()
        else:
            self.model.workflow_name = None
            self.workflow = None
        

    def setParam(self, node_name, param_name, value, index=None):
        if(self.workflow is not None):
            print(f"Setting param {node_name}.{param_name}[{index}] to {value}")
            self.workflow.setParam((node_name, param_name), value, index)


    def getSelectableInputNodes(self, param:NodeParameter) -> List[str]:
        selectable_nodes = []
        for node in selectable_nodes_config:
            node_return_type = inspect.signature(node.process).return_annotation
            print(f"Node return type: {node_return_type}")
            if(node_return_type == param.type):
                selectable_nodes.append(node.node_name)
        return selectable_nodes


    def createInputNode(self, param:NodeParameter, node_name):
        for node in selectable_nodes_config:
            if(node.node_name == node_name):
                param.value = copy.deepcopy(node)


    def runWorkflow(self):
        if(self.workflow is not None):
            self.workflowrunner.run(self.workflow, int(self.model.batch_size))
        else:
            print("No workflow loaded")

