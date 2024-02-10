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



class Controller:

    workflows = []
    workflow:FunctionalNode|None = None
    batch_size = 1
    workflowrunner = WorkflowRunner()

    def __init__(self):
        self.workflows = self.loadWorkflows()
        

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
            self.workflow = self.workflows[workflow_name].build()
        else:
            self.workflow = None
        
    def setParam(self, node_name, param_name, value):
        if(self.workflow is not None):
            self.workflow.setParam((node_name, param_name), value)

    def runWorkflow(self):
        print(self.batch_size)
        if(self.workflow is not None):
            self.workflowrunner.run(self.workflow, self.batch_size)
        else:
            print("No workflow loaded")


