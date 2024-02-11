from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.FunctionalTyping import ParamType
from diffuserslib.functional.WorkflowRunner import *
from PIL import Image
from dataclasses import dataclass
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
            workflows[module.name] = module
            print(f"Loaded workflow: {module.name}")
        return workflows
    
    def loadWorkflow(self, workflow_name):
        if workflow_name in self.workflows:
            self.model.workflow_name = workflow_name
            self.workflow = self.workflows[workflow_name].build()
        else:
            self.model.workflow_name = None
            self.workflow = None
        
    def setParam(self, node_name, param_name, value):
        if(self.workflow is not None):
            print(f"Setting param {node_name}.{param_name} to {value}")
            self.workflow.setParam((node_name, param_name), value)

    def runWorkflow(self):
        if(self.workflow is not None):
            self.workflowrunner.run(self.workflow, int(self.model.batch_size))
        else:
            print("No workflow loaded")


