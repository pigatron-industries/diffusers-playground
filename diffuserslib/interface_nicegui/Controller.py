from diffuserslib.functional import *
from diffuserslib.util import ModuleLoader
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
    workflows:Dict[str, WorkflowBuilder] = {}
    workflows_batch:Dict[str, str] = {}
    workflows_realtime:Dict[str, str] = {}
    workflows_sub:Dict[str, str] = {}
    workflow:FunctionalNode|None = None
    history_filename = ".history.yml"

    def __init__(self):
        if(WorkflowRunner.workflowrunner is not None):
            self.workflowrunner = WorkflowRunner.workflowrunner
        else:
            raise Exception("Workflow Runner not initialized")
        self.loadWorkflows()
        self.loadWorkflowParamsHistory()
        if('run_type' in self.workflow_history):
            self.model.run_type = self.workflow_history['run_type']
        if('workflow' in self.workflow_history):
            self.loadWorkflow(self.workflow_history['workflow'])
        self.loadSettings()
        

    def loadWorkflows(self):
        print("Loading workflow builders")
        self.workflows_batch = {}
        self.workflows_realtime = {}
        self.workflows_sub = {}
        path = os.path.join(os.path.dirname(__file__), '../functional_workflows')
        modules = ModuleLoader.load_from_directory(path)
        for module in modules:
            vars = ModuleLoader.get_vars(module)
            for name, builder in vars.items():
                if(issubclass(builder, WorkflowBuilder)):
                    workflow = builder()
                    self.workflows[name] = workflow
                    if(workflow.workflow and name not in self.workflows_batch):
                        self.workflows_batch[name] = workflow.name
                    if(workflow.subworkflow and name not in self.workflows_sub):
                        self.workflows_sub[name] = workflow.name
                    if(workflow.realtime and name not in self.workflows_realtime):
                        self.workflows_realtime[name] = workflow.name
                    print(f"Loading workflow builder: {name}")
        self.workflows_batch = dict(sorted(self.workflows_batch.items(), key=lambda item: item[1]))
        self.workflows_realtime = dict(sorted(self.workflows_realtime.items(), key=lambda item: item[1]))
        self.workflows_sub = dict(sorted(self.workflows_sub.items(), key=lambda item: item[1]))
        print(self.workflows_batch)
    

    def loadWorkflow(self, workflow_name):
        if(self.workflow is not None and self.workflow.name == workflow_name):
            return
        print(f"Loading workflow instance: {workflow_name}")
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


    def getSelectableInputNodes(self, param:NodeParameter) -> Dict[str, str]:
        selectable_subworkflows = {}
        for name in self.workflows_sub:
            workflow = self.workflows[name]
            if(workflow.type == param.type):
                selectable_subworkflows[name] = workflow.name
        return selectable_subworkflows


    def createInputNode(self, param:NodeParameter, workflow_name):
        if(workflow_name in self.workflows):
            param.value = self.workflows[workflow_name].build()
            param.value.node_name = workflow_name


    def runWorkflow(self):
        if(self.workflow is not None and WorkflowRunner.workflowrunner is not None):
            self.saveWorkflowParamsToHistory()
            WorkflowRunner.workflowrunner.run(self.workflow, int(self.model.batch_size))
        else:
            print("No workflow loaded")


    def getProgress(self) -> BatchProgressData|None:
        if(WorkflowRunner.workflowrunner is not None):
            return WorkflowRunner.workflowrunner.getProgress()


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
                print(paramstring)
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
            self.workflow_history['run_type'] = self.model.run_type
            self.saveWorkflowParamsHistory()


    def stopWorkflow(self):
        self.workflowrunner.stop()
        #  call interrupt when stop pressed a second time
        if(self.isStopping() and DiffusersPipelines.pipelines is not None):
            DiffusersPipelines.pipelines.interrupt()


    def removeRunData(self, batchid, runid):
        self.workflowrunner.batchrundata[batchid].rundata.pop(runid)
        self.workflowrunner.rundata.pop(runid)
        if(len(self.workflowrunner.batchrundata[batchid].rundata) == 0):
            self.workflowrunner.batchrundata.pop(batchid)


    def getWorkflowRunData(self):
        return self.workflowrunner.rundata
    

    def getBatchRunData(self):
        return self.workflowrunner.batchrundata
    

    def getBatchCurrentData(self) -> WorkflowBatchData|None:
        return self.workflowrunner.batchcurrent


    def getBatchQueueData(self):
        return self.workflowrunner.batchqueue


    def getWorkflowProgress(self):
        return self.workflowrunner.progress
    

    def isRunning(self):
        return self.workflowrunner.running


    def isStopping(self):
        return self.workflowrunner.stopping


    def saveOutput(self, key):
        self.workflowrunner.save(key, self.output_subdir)
        self.saveSettings()


    def saveSettings(self):
        self.workflow_history['output_subdir'] = self.output_subdir
        self.saveWorkflowParamsHistory()


    def loadSettings(self):
        if('output_subdir' in self.workflow_history):
            self.output_subdir = self.workflow_history['output_subdir']
        