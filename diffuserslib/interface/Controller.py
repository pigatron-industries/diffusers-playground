from diffuserslib.functional import FunctionalNode, WorkflowBuilder, WorkflowRunner, NodeParameter, BatchProgressData, WorkflowBatchData, UserInputNode
from diffuserslib.inference import DiffusersPipelines
from diffuserslib.util import ModuleLoader
from typing import Dict
from dataclasses import dataclass, field
from .Clipboard import Clipboard
import inspect
import yaml
import sys
import os


@dataclass
class Model:
    run_type:int = 1
    output_type:str = "Image"
    batch_size:int = 1
    workflow_name:str|None = None
    workflow:FunctionalNode|None = None
    workflows_sub:Dict[str, FunctionalNode] = field(default_factory=dict)

    

def str_to_class(str):
    return getattr(sys.modules[__name__], str)
    

class Controller:

    model = Model()
    builders:Dict[str, WorkflowBuilder] = {} # [WorkflowClass Name, WorkflowBuilder]
    builders_batch:Dict[str, str] = {}       # [WorkflowClass Name, Workflow Display Name]
    builders_realtime:Dict[str, str] = {}    # [WorkflowClass Name, Workflow Display Name]
    builders_sub:Dict[str, str] = {}         # [WorkflowClass Name, Workflow Display Name]
    output_types = ["Image", "Video", "Audio", "Text", "Other"]
    history_filename = ".history.yml"

    def __init__(self):
        if(WorkflowRunner.workflowrunner is not None):
            self.workflowrunner = WorkflowRunner.workflowrunner
        else:
            raise Exception("Workflow Runner not initialized")
        self.loadWorkflows()
        self.loadWorkflowParamsHistory()
        self.loadSettings()


    def filterWorkflowsByOutputType(self, workflows:Dict[str, str], typename:str) -> Dict[str, str]:
        filtered_workflows:Dict[str, str] = {}
        for name, display_name in workflows.items():
            builder = self.builders[name]
            if(name in self.builders):
                if(builder.type is not None and builder.type.__name__ == typename):
                    filtered_workflows[name] = display_name
                elif(builder.type is not None and typename == "Other" and builder.type.__name__ not in self.output_types):
                    filtered_workflows[name] = display_name
                elif(builder.type is None and typename == "Other"):
                    filtered_workflows[name] = display_name
        return filtered_workflows
        

    def loadWorkflows(self):
        print("Loading workflow builders")
        self.builders_batch = {}
        self.builders_realtime = {}
        self.builders_sub = {}
        path = os.path.join(os.path.dirname(__file__), '../functional_workflows')
        modules = ModuleLoader.load_from_directory(path)
        for module in modules:
            vars = ModuleLoader.get_vars(module)
            for name, buildercls in vars.items():
                if(inspect.isclass(buildercls) and issubclass(buildercls, WorkflowBuilder)):
                    builder = buildercls()
                    self.builders[name] = builder
                    if(builder.workflow and name not in self.builders_batch):
                        self.builders_batch[name] = builder.name
                    if(builder.subworkflow and name not in self.builders_sub):
                        self.builders_sub[name] = builder.name
                    if(builder.realtime and name not in self.builders_realtime):
                        self.builders_realtime[name] = builder.name
                    print(f"Loading workflow builder: {name}")
        self.builders_batch = dict(sorted(self.builders_batch.items(), key=lambda item: item[1]))
        self.builders_realtime = dict(sorted(self.builders_realtime.items(), key=lambda item: item[1]))
        self.builders_sub = dict(sorted(self.builders_sub.items(), key=lambda item: item[1]))
        print(self.builders_batch)
    

    def loadWorkflow(self, workflow_name):
        if(self.model.workflow is not None and self.model.workflow.name == workflow_name):
            return
        print(f"Loading workflow instance: {workflow_name}")
        self.model.workflows_sub = {}
        if workflow_name in self.builders:
            print(f"Loading workflow: {workflow_name}")
            self.model.workflow_name = workflow_name
            workflow_or_tuple = self.builders[workflow_name].build()
            if(isinstance(workflow_or_tuple, tuple)):
                self.model.workflow = workflow_or_tuple[0]
                secondary_workflows = workflow_or_tuple[1:]
                for secondary_workflow in secondary_workflows:
                    self.model.workflows_sub[secondary_workflow.name] = secondary_workflow
            else:
                self.model.workflow = workflow_or_tuple
            self.loadWorkflowParamsFromHistory()
            print(f"Loaded workflow: {self.model.workflow.name}")
            self.model.workflow.printDebug()
        else:
            self.model.workflow_name = None
            self.model.workflow = None


    def getSelectableInputNodes(self, param:NodeParameter) -> Dict[str, str]:
        selectable_subworkflows = {}
        for name in self.builders_sub:
            builder = self.builders[name]
            if(builder.type == param.type):
                selectable_subworkflows[name] = builder.name
        for name, workflow in self.model.workflows_sub.items():
            if(workflow.type == param.type):
                selectable_subworkflows[name] = workflow.name
        return selectable_subworkflows


    def createInputNode(self, param:NodeParameter, workflow_name):
        if(workflow_name in self.builders):
            param.value = self.builders[workflow_name].build()
            param.value.node_name = workflow_name
        elif(workflow_name in self.model.workflows_sub):
            param.value = self.model.workflows_sub[workflow_name]
            param.value.node_name = workflow_name


    def runWorkflow(self):
        if(self.model.workflow is not None and WorkflowRunner.workflowrunner is not None):
            self.saveWorkflowParamsToHistory()
            WorkflowRunner.workflowrunner.run(self.model.workflow, int(self.model.batch_size))
        else:
            print("No workflow loaded")

    
    def runSubWorkflow(self, workflow):
        if(WorkflowRunner.workflowrunner is not None):
            self.saveWorkflowParamsToHistory()
            WorkflowRunner.workflowrunner.run(workflow, 1)
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
        if(self.model.workflow is not None and self.model.workflow_name is not None and self.model.workflow_name in self.workflow_history):
            user_input_values = self.workflow_history[self.model.workflow_name]

            def visitor(param, parents):
                paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
                print(paramstring)
                if(paramstring+'.value' in user_input_values):
                    param.value.setValue(user_input_values[paramstring+'.value'])
                if(paramstring+'.node' in user_input_values):
                    self.createInputNode(param, user_input_values[paramstring+'.node'])
            
            self.model.workflow.visitParams(visitor)


    def saveWorkflowParamsToHistory(self):
        if(self.model.workflow is not None):
            user_input_values = {}

            def visitor(param, parents):
                paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
                if(isinstance(param.value, UserInputNode)):
                    user_input_values[paramstring+'.value'] = param.value.getValue()
                if(param.value != param.initial_value and isinstance(param.value, FunctionalNode) and isinstance(param.initial_value, UserInputNode)):
                    user_input_values[paramstring+'.node'] = param.value.node_name
                    
            self.model.workflow.visitParams(visitor)
            self.workflow_history[self.model.workflow_name] = user_input_values
            self.saveSettings()


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


    def copyOutput(self, runid):
        content = self.workflowrunner.rundata[runid].output
        if(content is not None):
            Clipboard.writeObject(content)


    def saveSettings(self):
        self.workflow_history['workflow'] = self.model.workflow_name
        self.workflow_history['run_type'] = self.model.run_type
        self.workflow_history['output_type'] = self.model.output_type
        self.workflow_history['output_subdir'] = self.output_subdir
        self.saveWorkflowParamsHistory()


    def loadSettings(self):
        if('output_subdir' in self.workflow_history):
            self.output_subdir = self.workflow_history['output_subdir']
        if('run_type' in self.workflow_history):
            self.model.run_type = self.workflow_history['run_type']
        if('output_type' in self.workflow_history):
            self.model.output_type = self.workflow_history['output_type']
        if(self.model.workflow is None and 'workflow' in self.workflow_history):
            self.loadWorkflow(self.workflow_history['workflow'])
        