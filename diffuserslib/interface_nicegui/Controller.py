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
    workflows = {}
    workflow:FunctionalNode|None = None
    history_filename = ".history.yml"

    def __init__(self):
        if(WorkflowRunner.workflowrunner is not None):
            self.workflowrunner = WorkflowRunner.workflowrunner
        else:
            raise Exception("Workflow Runner not initialized")
        self.loadWorkflows()
        self.loadWorkflowParamsHistory()
        if('workflow' in self.workflow_history):
            self.loadWorkflow(self.workflow_history['workflow'])
        

    def loadWorkflows(self):
        print("Loading workflows")
        path = os.path.join(os.path.dirname(__file__), '../functional_workflows')
        files = glob.glob(path + '/*.py')
        self.workflows = {}
        for file in files:
            spec = importlib.util.spec_from_file_location("module.workflow", file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module.workflow"] = module
            spec.loader.exec_module(module)
            self.workflows[module.name()] = module
    

    def loadWorkflow(self, workflow_name):
        print(f"Loading workflow: {workflow_name}")

        if(self.workflow is not None and self.workflow.name == workflow_name):
            return
        print(f"Loading workflow: {workflow_name}")
        if workflow_name in self.workflows:
            print(f"Loading workflow: {workflow_name}")
            self.model.workflow_name = workflow_name
            self.workflow = self.workflows[workflow_name].build()
            self.loadWorkflowParamsFromHistory()
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
            if(node_return_type == param.type):
                selectable_nodes.append(node.node_name)
        return selectable_nodes


    def createInputNode(self, param:NodeParameter, node_name):
        for node in selectable_nodes_config:
            if(node.node_name == node_name):
                param.value = copy.deepcopy(node)


    def runWorkflow(self):
        if(self.workflow is not None and WorkflowRunner.workflowrunner is not None):
            self.saveWorkflowParamsToHistory()
            WorkflowRunner.workflowrunner.run(self.workflow, int(self.model.batch_size))
        else:
            print("No workflow loaded")


    def loadWorkflowParamsHistory(self):
        if os.path.exists(self.history_filename):
            file = open(self.history_filename, "r")
            self.workflow_history = yaml.full_load(file)
            if(self.workflow_history is None):
                self.workflow_history = {}
        else:
            self.workflow_history = {}

    
    def saveWorkflowParamsHistory(self):
        file = open(self.history_filename, "w")
        yaml.dump(self.workflow_history, file)


    def loadWorkflowParamsFromHistory(self):
        if(self.workflow is not None and self.model.workflow_name is not None and self.model.workflow_name in self.workflow_history):
            user_input_values = self.workflow_history[self.model.workflow_name]

            def visitor(param, parents):
                paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
                if(paramstring+'.value' in user_input_values):
                    param.value.setValue(user_input_values[paramstring+'.value'])
                if(paramstring+'.node' in user_input_values):
                    self.createInputNode(param, user_input_values[paramstring+'.node'])
            
            self.workflow.visitParams(visitor)


    def saveWorkflowParamsToHistory(self):
        if(self.workflow is not None):
            user_input_values = {}

            def visitor(param, parents):
                paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
                if(isinstance(param.value, UserInputNode)):
                    user_input_values[paramstring+'.value'] = param.value.getValue()
                if(param.value != param.initial_value and isinstance(param.value, FunctionalNode) and isinstance(param.initial_value, UserInputNode)):
                    user_input_values[paramstring+'.node'] = param.value.node_name
                    
            self.workflow.visitParams(visitor)
            self.workflow_history[self.model.workflow_name] = user_input_values
            self.workflow_history['workflow'] = self.model.workflow_name
            self.saveWorkflowParamsHistory()


    def stopWorkflow(self):
        self.workflowrunner.stop()


    def getWorkflowRunData(self):
        return self.workflowrunner.rundata
    

    def getWorkflowProgress(self):
        return self.workflowrunner.progress
    

    def isRunning(self):
        return self.workflowrunner.running


    def isStopping(self):
        return self.workflowrunner.stopping


    def saveOutput(self, key):
        self.workflowrunner.save(key)
